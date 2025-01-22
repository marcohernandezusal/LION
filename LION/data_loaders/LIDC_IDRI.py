# =============================================================================
# This file is part of LION library
# License : BSD-3
#
# Author  : Emilien Valat
# Modifications: Michelle Limbach, Ander Biguri, Marco Hernández
# =============================================================================


from typing import List, Dict
import pathlib
import random
import math

import torch
import numpy as np
import json
import os
from typing import List, Dict
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import label, binary_fill_holes, binary_dilation, binary_erosion, zoom
from scipy.signal import welch
from scipy.fft import fft, ifft

from LION.utils.paths import LIDC_IDRI_PROCESSED_DATASET_PATH
import LION.CTtools.ct_utils as ct
from LION.utils.parameter import LIONParameter

import logging

logging.basicConfig(filename="logs/volume_debug.log", level=logging.DEBUG, format="%(asctime)s - %(message)s")


def format_index(index: int) -> str:
    str_index = str(index)
    while len(str_index) < 4:
        str_index = "0" + str_index
    assert len(str_index) == 4
    return str_index


def load_json(file_path: pathlib.Path):
    if not file_path.is_file():
        raise FileNotFoundError(f"No file found at {file_path}")
    return json.load(open(file_path))


def choose_random_annotation(
    nodule_annotations_list: List,
) -> str:
    return random.choice(nodule_annotations_list)


def create_consensus_annotation(
    path_to_patient_folder: pathlib.Path,
    slice_index: int,
    nodule_index: str,
    nodule_annotations_list: List,
    clevel: float,
) -> torch.int16:
    masks = []
    if isinstance(path_to_patient_folder, str):
        path_to_patient_folder = pathlib.Path(path_to_patient_folder)
    for annotation in nodule_annotations_list:
        path_to_mask = path_to_patient_folder.joinpath(
            f"mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation}.npy"
        )
        current_annotation_mask = np.load(path_to_mask)
        masks.append(current_annotation_mask)

    nodule_mask = torch.from_numpy(np.mean(masks, axis=0) >= clevel)
    return nodule_mask


class LIDC_IDRI(Dataset):
    def __init__(
        self,
        mode,
        geometry_parameters: ct.Geometry = None,
        parameters: LIONParameter = None,
    ):
        """
        Initializes LIDC-IDRI dataset.

        Parameters:
            - device (torch.device): Selects the device to use for the data loader.
            - task (str): Defines pipeline on how to use data. Distinguish between "joint", "end_to_end", "segmentation", "reconstruction" and "diagnostic".
                          Dataset will return, for each task:
                          "segmentation"    -> (image, segmentation_label)
                          "reconstruction"  -> (sinogram, image_label)
                          "diagnostic"      -> (segmented_nodule, diagnostic_label)
                          "joint"           -> ?????
                          "end_to_end"      -> ?????
            - training_proportion (float): Defines training % of total data.
            - mode (str): Defines "train", "validation" or "test" mode.
            - Task (str): Defines what task is the Dataset being used for. "segmentation" (default) returns (gt_image,segmentation) pairs while "reconstruction" returns (sinogram, gt_image) pairs
            - annotation (str): Defines what annotation mode to use. Distinguish between "random" and "consensus". Default "consensus"
            - max_num_slices_per_patient (int): Defines the maximum number of slices to take per patient. Default is -1, which takes all slices we have of each patient and pcg_slices_nodule gets ignored.
            - pcg_slices_nodule (float): Defines percentage of slices with nodule in dataset. 0 meaning "no nodules at all" and 1 meaning "just take slices that contain annotated nodules". Only used if max_num_slices_per_patient != -1. Default is 0.5.
            - clevel (float): Defines consensus level if annotation=consensus. Value between 0-1. Default is 0.5.
            - geo: Geometry() type, if sinograms are requied (e.g. fo "reconstruction")
            - lung_only (bool): If True, processes each slice to return only the lung regions.

        """

        # Input parsing
        assert mode in [
            "train",
            "validation",
            "test",
        ], f'Wrong mode argument, must be in ["train", "validation", "test"]'

        if parameters is None:
            parameters = LIDC_IDRI.default_parameters()
        self.params = parameters

        task = self.params.task
        assert task in [
            "joint",
            "end_to_end",
            "segmentation",
            "reconstruction",
            "diagnostic",
        ], f'task argument {task} not in ["joint", "end_to_end", "segmentation", "reconstruction", "diagnostic"]'

        if task not in ["segmentation", "reconstruction", "end_to_end"]:
            raise NotImplementedError(f"task {task} not implemented yet")

        if (
            task in ["reconstruction"]
            and geometry_parameters is None
            and self.params.geo is None
        ):
            raise ValueError("geo input required for recosntruction modes")

        # Aux variable setting
        self.sinogram_transform = None
        self.image_transform = None
        self.device = self.params.device

        if task in ["reconstruction"]:
            self.image_transform = ct.from_HU_to_mu
        if task in ["segmentation"]:
            self.image_transform = ct.from_HU_to_normal

        if geometry_parameters is not None:
            self.params.geo = geometry_parameters
            self.operator = ct.make_operator(geometry_parameters)
        elif self.params.geo is not None:
            self.operator = ct.make_operator(self.params.geo)


        # Start of Patient pre-processing

        self.path_to_processed_dataset = pathlib.Path(self.params.folder)
        self.patients_masks_dictionary = load_json(
            self.path_to_processed_dataset.joinpath("patients_masks.json")
        )

        # Add patients without masks, for now hardcoded, find a solution in preprocessing
        self.patients_masks_dictionary["LIDC-IDRI-0238"] = {}
        self.patients_masks_dictionary["LIDC-IDRI-0585"] = {}

        self.patients_diagnosis_dictionary = load_json(
            self.path_to_processed_dataset.joinpath("patient_id_to_diagnosis.json")
        )
        self.total_patients = (
            len(list(self.path_to_processed_dataset.glob("LIDC-IDRI-*"))) + 1
        )
        self.num_slices_per_patient = self.params.max_num_slices_per_patient
        self.pcg_slices_nodule = self.params.pcg_slices_nodule
        self.annotation = self.params.annotation
        self.clevel = (
            self.params.clevel
        )  # consensus level, only used if annotation == consensus
        self.patient_index_to_n_slices_dict: Dict = {
            f"LIDC-IDRI-{format_index(index)}": len(
                list(
                    self.path_to_processed_dataset.joinpath(
                        f"LIDC-IDRI-{format_index(index)}"
                    ).glob("slice_*.npy")
                )
            )
            for index in range(1, self.total_patients)
        }

        # Dict with all slices of each patient
        self.patient_index_to_slices_index_dict: Dict = {
            f"LIDC-IDRI-{format_index(index)}": list(
                np.arange(
                    0,
                    self.patient_index_to_n_slices_dict[
                        f"LIDC-IDRI-{format_index(index)}"
                    ],
                    1,
                )
            )
            for index in range(1, self.total_patients)
        }

        # Dict with all nodule slices of each patient
        # Converts the keys from self.patients_masks_dictionary to integer
        self.patient_index_to_nodule_slices_index_dict: Dict = {
            f"LIDC-IDRI-{format_index(index)}": [
                int(item)
                for item in list(
                    self.patients_masks_dictionary[
                        f"LIDC-IDRI-{format_index(index)}"
                    ].keys()
                )
            ]
            for index in range(1, self.total_patients)
        }

        # Dict with all non-nodule slices of each patient
        # Computes as difference of all slices dict and dict with nodules
        self.patient_index_to_non_nodule_slices_index_dict: Dict = {
            f"LIDC-IDRI-{format_index(index)}": list(
                set(
                    self.patient_index_to_slices_index_dict[
                        f"LIDC-IDRI-{format_index(index)}"
                    ]
                )
                - set(
                    self.patient_index_to_nodule_slices_index_dict[
                        f"LIDC-IDRI-{format_index(index)}"
                    ]
                )
            )
            for index in range(1, self.total_patients)
        }

        # Corrupted data handling
        # Delete all slices that contain a nodule that has more than 4 annotations
        self.removed_slices: Dict = {}
        for (
            patient_id,
            nodule_slices_list,
        ) in self.patient_index_to_nodule_slices_index_dict.items():
            self.removed_slices[patient_id] = []
            for slice_index in nodule_slices_list:
                all_nodules_dict: Dict = self.patients_masks_dictionary[patient_id][
                    f"{slice_index}"
                ]
                for _, nodule_annotations_list in all_nodules_dict.items():
                    if len(nodule_annotations_list) > 4:
                        self.removed_slices[patient_id].append(slice_index)
                        break

        self.patient_index_to_nodule_slices_index_dict: Dict = {
            f"LIDC-IDRI-{format_index(index)}": list(
                set(
                    self.patient_index_to_nodule_slices_index_dict[
                        f"LIDC-IDRI-{format_index(index)}"
                    ]
                )
                - set(self.removed_slices[f"LIDC-IDRI-{format_index(index)}"])
            )
            for index in range(1, self.total_patients)
        }

        ##% Divide dataset in training/validation/testing
        self.training_proportion = self.params.training_proportion
        self.validation_proportion = self.params.validation_proportion
        self.params.mode = mode
        # Commpute number if images for each
        self.n_patients_training = math.floor(
            self.training_proportion * (self.total_patients)
        )
        self.n_patients_validation = math.floor(
            self.validation_proportion * (self.total_patients)
        )
        self.n_patients_testing = (
            self.total_patients - self.n_patients_training - self.n_patients_validation
        )

        assert self.total_patients == (
            self.n_patients_training
            + self.n_patients_testing
            + self.n_patients_validation
        ), print(
            f"Total patients: {self.total_patients}, \n training patients {self.n_patients_training}, \n validation patients {self.n_patients_validation}, \n testing patients {self.n_patients_testing}"
        )

        # Get patient IDs for each
        self.patient_ids = list(self.patient_index_to_n_slices_dict.keys())
        self.training_patients_list = self.patient_ids[: self.n_patients_training]
        self.validation_patients_list = self.patient_ids[
            self.n_patients_training : self.n_patients_training
            + self.n_patients_validation
        ]
        self.testing_patients_list = self.patient_ids[
            self.n_patients_training + self.n_patients_validation :
        ]
        # print("Training patients:", self.training_patients_list)
        # print("Validation patients:", self.validation_patients_list)
        # print("Testing patients:", self.testing_patients_list)
        assert len(self.patient_ids) == len(self.training_patients_list) + len(
            self.testing_patients_list
        ) + len(self.validation_patients_list), print(
            f"Len patients ids: {len(self.patient_ids)}, \n len training patients {len(self.training_patients_list)},\n len validation patients {len(self.validation_patients_list)}, \n len testing patients {len(self.testing_patients_list)}"
        )

        logging.info(f"Total patients: {self.total_patients}")
        logging.info(f"Training patients: {self.n_patients_training}")
        logging.info(f"Validation patients: {self.n_patients_validation}")
        logging.info(f"Testing patients: {self.n_patients_testing}")

            

        print("Preparing patient list, this may take time....")
        if self.params.mode == "train":
            patient_list_to_load = self.training_patients_list
        elif self.params.mode == "validation":
            patient_list_to_load = self.validation_patients_list
        elif self.params.mode == "test":
            patient_list_to_load = self.testing_patients_list
        else:
            raise NotImplementedError(
                f"mode {self.params.mode} not implemented, try training, validation or testing"
            )
        # print(f"Dataset initialized with mode: {self.params.mode}")
        # print(f"Patients to load for mode {self.params.mode}: {patient_list_to_load}")

        # Load tumor metadata if min_tumor_size or max_tumor_size is specified
        if self.params.min_tumor_size is not None or self.params.max_tumor_size is not None:
            metadata_path = f"tumor_metadata_files/{mode}_tumor_metadata.json"
            with open(metadata_path, "r") as f:
                tumor_metadata = json.load(f)

            # Define filtering criteria
            min_tumor_size = self.params.min_tumor_size or 0
            max_tumor_size = self.params.max_tumor_size or float("inf")

            # Filter patients based on tumor size
            def filter_patients(patient_list):
                return [
                    patient_id
                    for patient_id in patient_list
                    if min_tumor_size
                    <= tumor_metadata.get(patient_id, {}).get("max_tumor_axis_length", 0)
                    <= max_tumor_size
                ]

            if mode == "train":
                self.training_patients_list = filter_patients(self.training_patients_list)
                logging.info(f"Filtered {mode} patient list: {len(self.training_patients_list)} patients remaining")
            elif mode == "validation":
                self.validation_patients_list = filter_patients(self.validation_patients_list)
                logging.info(f"Filtered {mode} patient list: {len(self.validation_patients_list)} patients remaining")
            elif mode == "test":
                self.testing_patients_list = filter_patients(self.testing_patients_list)
                logging.info(f"Filtered {mode} patient list: {len(self.testing_patients_list)} patients remaining")

        
        self.slices_to_load = self.get_slices_to_load(
            patient_list_to_load,
            self.patient_index_to_non_nodule_slices_index_dict,
            self.patient_index_to_nodule_slices_index_dict,
            self.num_slices_per_patient,
            self.pcg_slices_nodule,
        )
        self.slice_index_to_patient_id_list = self.get_slice_index_to_patient_id_list(
            self.slices_to_load
        )
        self.patient_id_to_first_index_dict = self.get_patient_id_to_first_index_dict(
            self.slices_to_load
        )

        print(f"Patient lists ready for {self.params.mode} dataset")

        if task == "segmentation" and self.params.lung_only:
            self.preprocess_pipeline_setup()

        if self.params.volume_representation:
            self.voxel_spacing = self.params.voxel_spacing
            self.sampling_thickness = self.params.sampling_thickness or {}
            self.interpolation_method = self.params.interpolation_method
            self.bspline_order = self.params.bspline_order
            self.sinc_width = self.params.welch_sinc_width

        self.patient_list_to_load = patient_list_to_load

        # ============== FINAL CHECK to remove zero-tumor patients =============
        self._final_check_and_filter_patients(mode)
        # =======================================================================


    def _final_check_and_filter_patients(self, mode: str):
        """
        Final pass: For each patient in the relevant list, load the slices in minimal form
        and check if the tumor mask is completely empty after real filtering logic.
        If it is empty, log that the patient does not match the metadata but do NOT remove them.
        """
        if mode == "train":
            patient_list = self.training_patients_list
        elif mode == "validation":
            patient_list = self.validation_patients_list
        elif mode == "test":
            patient_list = self.testing_patients_list
        else:
            return  # no-op for safety

        for pid in patient_list:
            volume_and_mask = self.get_patient_volume(pid)

            # If get_patient_volume returned None or the mask is entirely zeros, log mismatch
            if volume_and_mask is None:
                logging.warning(
                    f"Patient {pid} in mode={mode} yields no valid volume/mask "
                    f"(metadata said there should be a tumor)."
                )
                continue

            # Check for tumor_patch_mode
            if isinstance(volume_and_mask, list):
                # It's a list of patches
                if len(volume_and_mask) == 0:
                    logging.warning(
                        f"Patient {pid} in mode={mode} yields 0 patches "
                        f"(metadata said there should be a tumor)."
                    )
            else:
                # It's (volume_tensor, mask_tensor)
                volume_tensor, mask_tensor = volume_and_mask
                # Check if there are any nonzero tumor voxels
                if mask_tensor[1].sum() == 0:
                    logging.warning(
                        f"Patient {pid} in mode={mode} yields an empty tumor mask "
                        f"(metadata said there should be a tumor)."
                    )

        logging.info(f"Final check complete for mode={mode}. Potential mismatches have been logged.")

    @staticmethod
    def default_parameters(geo=None, task="reconstruction"):
        param = LIONParameter()
        param.name = "LIDC-IDRI Data Loader"
        param.training_proportion = 0.8
        param.validation_proportion = 0.1
        param.testing_proportion = (
            1 - param.training_proportion - param.validation_proportion
        )  # not used, but for metadata
        param.max_num_slices_per_patient = 5
        param.pcg_slices_nodule = 0.5
        param.task = task
        param.folder = LIDC_IDRI_PROCESSED_DATASET_PATH
        if task == "reconstruction" and geo is None:
            raise ValueError(
                "For reconstruction task geometry needs to be input to default_parameters(geo=geometry_param)"
            )

        # segmentation specific
        param.clevel = 0.5
        param.annotation = "consensus"
        param.device = torch.cuda.current_device()
        param.geo = geo

        param.lung_only = False
        param.normalize_lungs = "none"  # Options: "none", "minmax", "zscore"

        # parameters for volume representation
        param.volume_representation = False
        param.voxel_spacing = (1.0, 1.0, 1.0)  # Default target voxel spacing
        param.sampling_thickness = {}  # Default to empty; can be set later
        param.interpolation_method = 'bspline' # Default interpolation method
        param.bspline_order = 3 # Default B-spline interpolation order
        param.welch_sinc_width = 5 # Default Welch window sinc width
        # Set tumor size filtering parameters
        param.min_tumor_size = None
        param.max_tumor_size = None
        # Set patch parameters
        param.tumor_patch_mode = False  # Default: return full images
        param.patch_size = (80, 80)  # Patch size (width, height)

        return param

    def get_slices_to_load(
        self,
        patient_list: List,
        non_nodule_slices_dict: Dict,
        nodule_slices_dict: Dict,
        num_slices_per_patient: int,
        pcg_slices_nodule: float,
    ):
        """
        Returns a dictionary that contains patient_id's as keys and list of slices to load as values for each patient.

        Parameters:
            - patient_list (List): List that contains patient_id of all patients.
            - non_nodule_slices_dict (Dict): Dict that contains all slices without nodule of each patient_id.
            - nodule_slices_dict (Dict): Dict that contains all slices with nodule of each patient_id.
            - num_slices_per_patient (int): Defines maximum number of slices we want per patient. If num_slices_per_patient=-1 take all slices we have of each patient.
            - pcg_slices_nodule (float): Defines amount of slices that should contain a nodule. Value between 0-1.
        Returns:
            - patient_id_to_slices_to_load_dict which contains patient_id as key and list of slices to load as values
        """
        patient_id_to_slices_to_load_dict = {}

        if num_slices_per_patient == -1:
            num_slices_per_patient = 1000

        for patient_id in patient_list:  # Loop over every patient
            number_of_slices = min(
                num_slices_per_patient,
                min(
                    len(non_nodule_slices_dict[patient_id]),
                    len(nodule_slices_dict[patient_id]),
                ),
            )

            # Get amount of slices we want without nodule
            number_of_slices_without_nodule = int(
                np.ceil(number_of_slices * (1 - pcg_slices_nodule))
            )
            # Get amount of slices we want with nodule
            number_of_slices_with_nodule = (
                number_of_slices - number_of_slices_without_nodule
            )

            # Get linspace of non-nodule and nodule slices of each patient and afterwards sort the list in increasing order
            patient_id_to_slices_to_load_dict[patient_id] = list(
                np.array(non_nodule_slices_dict[patient_id])[
                    np.linspace(
                        0,
                        len(non_nodule_slices_dict[patient_id]),
                        number_of_slices_without_nodule,
                        dtype=int,
                        endpoint=False,
                    )
                ]
            ) + list(
                np.array(nodule_slices_dict[patient_id])[
                    np.linspace(
                        0,
                        len(nodule_slices_dict[patient_id]),
                        number_of_slices_with_nodule,
                        dtype=int,
                        endpoint=False,
                    )
                ]
            )
            patient_id_to_slices_to_load_dict[patient_id].sort()
            

        return patient_id_to_slices_to_load_dict

    def get_patient_id_to_first_index_dict(self, patient_with_slices_to_load: Dict):
        """
        Returns a dictionary that contains patient_id's as keys and start index of each patient in self.slice_index_to_patient_id_list as value.

        Parameters:
            - patient_with_slices_to_load (Dict): Dict that defines which slices to load per patient.
        Returns:
            - patient_id_to_first_index_dict (Dict): Defines start index of each patient in self.slice_index_to_patient_id_list. Needed for mapping of global index to slice index.
        """
        patient_id_to_first_index_dict = {}
        global_index = 0
        for patient_id in patient_with_slices_to_load:
            path_to_folder = self.path_to_processed_dataset.joinpath(patient_id)
            patient_id_to_first_index_dict[patient_id] = global_index

            if len(patient_with_slices_to_load[patient_id]) < len(
                list(path_to_folder.glob("slice_*.npy"))
            ):
                global_index += len(patient_with_slices_to_load[patient_id])
            else:
                global_index += len(list(path_to_folder.glob("slice_*.npy")))
        return patient_id_to_first_index_dict

    def get_slice_index_to_patient_id_list(self, patient_with_slices_to_load: Dict):
        """
        Returns a list that contains "number of slices" times each patient id.

        Parameters:
            - patient_with_slices_to_load (Dict): Dict that defines which slices to load per patient.
        Returns:
            - slice_index_to_patient_id_list (List): Contains number of slices times each patient id. Needed for mapping of global index to slice index.
        """
        slice_index_to_patient_id_list = []
        for patient_id in patient_with_slices_to_load:
            path_to_folder = self.path_to_processed_dataset.joinpath(patient_id)

            if len(patient_with_slices_to_load[patient_id]) < len(
                list(path_to_folder.glob("slice_*.npy"))
            ):
                n_slices = len(patient_with_slices_to_load[patient_id])
            else:
                n_slices = len(list(path_to_folder.glob("slice_*.npy")))

            for slice_index in range(n_slices):
                slice_index_to_patient_id_list.append(patient_id)
        return slice_index_to_patient_id_list

    def get_reconstruction_tensor(self, file_path: pathlib.Path) -> torch.Tensor:
        tensor = torch.from_numpy(np.load(file_path)).unsqueeze(0).to(self.device)
        return tensor

    def set_sinogram_transform(self, sinogram_transform):
        self.sinogram_transform = sinogram_transform

    def set_image_transform(self, image_transform):
        self.image_transform = image_transform

    def compute_clean_sinogram(self, image=None) -> torch.Tensor:

        if self.operator is None:
            raise AttributeError("CT operator not know. Have you given a ct geometry?")
        sinogram = self.operator(image)
        return sinogram

    def filter_tumors_by_size(self, mask_volume: np.ndarray, min_size: float = None, max_size: float = None) -> np.ndarray:
        """
        Filters tumors in a 3D mask by removing tumors whose longest axis is outside the specified range.

        Parameters:
            mask_volume (np.ndarray): 3D binary mask of the tumors (depth, height, width).
            min_size (float): Minimum tumor size to keep (in mm). Default is None.
            max_size (float): Maximum tumor size to keep (in mm). Default is None.

        Returns:
            np.ndarray: Filtered 3D mask with tumors within the specified size range.
        """
        labeled_mask, num_features = label(mask_volume)  # Label connected components (tumors)
        
        # If no tumors are found, return the original mask
        if num_features == 0:
            return mask_volume

        # Initialize an empty mask to store filtered tumors
        filtered_mask = np.zeros_like(mask_volume)

        # Loop over each labeled tumor
        for tumor_label in range(1, num_features + 1):
            tumor = (labeled_mask == tumor_label)  # Extract the current tumor as a binary mask
            
            # Calculate the bounding box of the tumor
            coords = np.argwhere(tumor)
            min_coords = np.min(coords, axis=0)
            max_coords = np.max(coords, axis=0)
            
            # Compute the longest axis of the tumor
            longest_axis = np.linalg.norm(max_coords - min_coords)
            logging.debug(f"Tumor {tumor_label} has longest axis of {longest_axis:.2f} mm")
            
            # Check if the tumor is within the specified size range
            if ((min_size is None or longest_axis >= min_size) and
                (max_size is None or longest_axis <= max_size)):
                filtered_mask = np.logical_or(filtered_mask, tumor)  # Keep the tumor
        
        return filtered_mask

    def get_mask_tensor(self, patient_id: str, slice_index: int) -> torch.Tensor:
        """
        Returns the segmentation mask for a given patient and slice index.
        If any mask file is missing, it returns an empty mask.
        """
        try:
            # Initialize an empty mask for the entire slice
            mask = torch.zeros((512, 512), dtype=torch.bool)
            
            # Attempt to retrieve the annotation information for the given slice
            all_nodules_dict: Dict = self.patients_masks_dictionary[patient_id][f"{slice_index}"]
            
            for nodule_index, nodule_annotations_list in all_nodules_dict.items():
                nodule_masks = []

                for annotation in nodule_annotations_list:
                    # Construct path for each annotation
                    path_to_mask = self.path_to_processed_dataset.joinpath(
                        f"{patient_id}/mask_{slice_index}_nodule_{nodule_index}_annotation_{annotation}.npy"
                    )

                    # Check if the mask file exists
                    if os.path.isfile(path_to_mask):
                        current_annotation_mask = np.load(path_to_mask)
                        nodule_masks.append(current_annotation_mask)
                    # else:
                    #     print(f"Warning: Mask file {path_to_mask} not found for annotation {annotation}. Skipping.")

                # Aggregate masks for the current nodule if any were found
                if nodule_masks:
                    nodule_mask = torch.from_numpy(np.mean(nodule_masks, axis=0) >= self.clevel)
                    mask = mask.bitwise_or(nodule_mask)

        except KeyError:
            # If no annotations are found for the given slice, use an empty mask
            # print(f"Warning: No annotations found for slice {slice_index} of patient {patient_id}. Using empty mask.")
            mask = torch.zeros((512, 512), dtype=torch.bool)

        # Create the background as the inverse of the mask
        background = ~mask
        return torch.stack((background, mask))

    def __len__(self):
        if self.params.volume_representation:
            return len(self.patient_list_to_load)  # Number of patients
        else:
            return len(self.slice_index_to_patient_id_list)  # Number of slices

    def get_specific_slice(self, patient_index, slice_index):
        ## Assumes slice and mask exist
        file_path = self.path_to_processed_dataset.joinpath(
            f"{patient_index}/slice_{slice_index}.npy"
        )
        return self.get_reconstruction_tensor(file_path), self.get_mask_tensor(
            patient_index, slice_index
        )

    def interpolate_volume(self, volume, original_spacing, target_spacing, method = 'bspline', order = 3, sinc_width = 5):
        """
        Resample a 3D volume to the target spacing using the specified interpolation method.

        Parameters:
            - volume (np.ndarray): Input 3D volume (z, y, x).
            - original_spacing (tuple): Original voxel spacing (z, y, x).
            - target_spacing (tuple): Desired voxel spacing (z, y, x).
            - method (str): Interpolation method. Options: 'bspline', 'welch_sinc'.
            - order (int): B-Spline order. Relevant if method='bspline'. Default is 3 (cubic).
            - sinc_width (int): Number of sinc lobes on each side. Relevant if method='welch_sinc'. Default is 5.

        Returns:
            - Resampled 3D volume (np.ndarray).
        """
        assert len(original_spacing) == volume.ndim, (
            f"Original spacing {original_spacing} must match volume dimensions {volume.ndim}."
        )
        assert len(target_spacing) == volume.ndim, (
            f"Target spacing {target_spacing} must match volume dimensions {volume.ndim}."
        )
        assert method in ['bspline', 'welch_sinc'], (
            f"Unsupported interpolation method '{method}'. Choose 'bspline' or 'welch_sinc'."
        )

        resize_factors = [original / target for original, target in zip(original_spacing, target_spacing)]
        logging.debug(f"Interpolation method: {method}")
        logging.debug(f"Original spacing: {original_spacing}, Target spacing: {target_spacing}")
        logging.debug(f"Resize factors: {resize_factors}")

        if method == 'bspline':
            logging.debug(f"Performing B-Spline interpolation with order={order}")
            resampled = zoom(volume, resize_factors, order=order)
            logging.debug(f"Resampled volume shape (B-Spline): {resampled.shape}")
            return resampled

        elif method == 'welch_sinc':
            logging.debug(f"Performing Welch windowed sinc interpolation with sinc_width={sinc_width}")
            resampled = self.welch_windowed_sinc_interpolate(volume, resize_factors, sinc_width)
            logging.debug(f"Resampled volume shape (Welch Sinc): {resampled.shape}")
            return resampled

    def welch_windowed_sinc_interpolate(
        self,
        volume: np.ndarray,
        resize_factors: list,
        sinc_width: int = 5
    ) -> np.ndarray:
        """
        Apply Welch windowed sinc interpolation to a 3D volume.

        Parameters:
            - volume (np.ndarray): 3D volume.
            - resize_factors (list): Resize factors for each axis.
            - sinc_width (int): Number of sinc lobes on each side.

        Returns:
            - Resampled 3D volume (np.ndarray).
        """
        logging.debug("Starting Welch windowed sinc interpolation.")

        def resample_axis(data, axis, factor, sinc_width):
            """
            Resample data along a single axis using windowed sinc interpolation.

            Parameters:
                - data (np.ndarray): Input data.
                - axis (int): Axis to resample.
                - factor (float): Resize factor.
                - sinc_width (int): Number of sinc lobes on each side.

            Returns:
                - Resampled data.
            """
            logging.debug(f"Resampling axis {axis} with factor {factor} and sinc_width {sinc_width}.")

            original_size = data.shape[axis]
            target_size = int(np.round(original_size * factor))
            logging.debug(f"Original size: {original_size}, Target size: {target_size}")

            # Create the output indices
            target_indices = np.linspace(0, original_size, target_size, endpoint=False)

            # Define the sinc kernel
            # Calculate the distance between original and target samples
            x = (target_indices - np.arange(original_size)[:, np.newaxis])
            x = x / factor  # Scale by resize factor
            x = x[np.abs(x) <= sinc_width]  # Limit to sinc_width

            # Apply sinc function
            sinc_kernel = np.sinc(x)

            # Apply Welch (Hanning) window
            window = get_window('hanning', sinc_kernel.shape[1])
            windowed_sinc = sinc_kernel * window
            windowed_sinc /= np.sum(windowed_sinc, axis=1, keepdims=True)  # Normalize

            # Perform convolution using the windowed sinc kernel
            resampled = convolve1d(data, windowed_sinc, axis=axis, mode='mirror')

            logging.debug(f"Completed resampling axis {axis}.")
            return resampled

        resampled_volume = volume.copy()
        for axis, factor in enumerate(resize_factors):
            if factor == 1.0:
                logging.debug(f"No resampling needed for axis {axis}. Skipping.")
                continue  # Skip axes with no resampling needed
            resampled_volume = resample_axis(resampled_volume, axis, factor, sinc_width)

        logging.debug("Completed Welch windowed sinc interpolation.")
        return resampled_volume


    def normalize_lungs(self, volume, method="none"):
        """
        Normalize a 3D volume based on the specified method.

        Parameters:
        - volume (np.ndarray): Input 3D volume (z, y, x).
        - method (str): Normalization method. Options: "none", "minmax", "zscore".

        Returns:
        - np.ndarray: Normalized volume.
        """
        if method == "none":
            return volume
        elif method == "minmax":
            volume_min, volume_max = volume.min(), volume.max()
            return (volume - volume_min) / (volume_max - volume_min)
        elif method == "zscore":
            mean, std = volume.mean(), volume.std()
            return (volume - mean) / (std + 1e-8)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    def extract_tumor_centered_patches(self, volume, mask, patch_size=(80, 80)):
        """
        Extracts patches centered on each tumor centroid.

        Parameters:
            volume (np.ndarray): 3D volume array of shape (depth, height, width).
            mask (np.ndarray): 3D binary mask of the tumors (depth, height, width).
            patch_size (tuple): Size of the patch (width, height).

        Returns:
            list of tuples: Each tuple contains a patch and the corresponding mask patch.
        """
        depth, height, width = volume.shape
        mask = mask[1,:,:,:]
        patch_half_size = (patch_size[0] // 2, patch_size[1] // 2)
        labeled_mask, num_features = label(mask)
        logging.debug(f"Volume and mask shapes: {volume.shape}, {mask.shape} in extract_tumor_centered_patches")
        
        patches = []
        for tumor_label in range(1, num_features + 1):
            coords = np.argwhere(labeled_mask == tumor_label)
            centroid = np.mean(coords, axis=0).astype(int)
            z, y, x = centroid

            # Compute patch boundaries
            y_start, y_end = max(0, y - patch_half_size[0]), min(height, y + patch_half_size[0])
            x_start, x_end = max(0, x - patch_half_size[1]), min(width, x + patch_half_size[1])
            
            # Calculate actual start and end indices within image bounds
            y_start_clipped, y_end_clipped = max(0, y - patch_half_size[0]), min(height, y + patch_half_size[0])
            x_start_clipped, x_end_clipped = max(0, x - patch_half_size[1]), min(width, x + patch_half_size[1])

            volume_patch_list = []
            mask_patch_list = []
            # Iterate through the slices (z) where the tumor exists in the mask
            for slice in range(coords[0][0], coords[-1][0] + 1):
            # Extract patch within valid bounds
                logging.debug(f"Extracting patch centered at ({z}, {y}, {x})")
                logging.debug(f"Volume shape: {volume.shape}, Mask shape: {mask.shape}")
                patch_volume = volume[slice, y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped]
                patch_mask = mask[slice, y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped]

                # Calculate padding required for each side
                top_padding = max(0, patch_half_size[0] - y_start)
                bottom_padding = max(0, y_end - (height - patch_half_size[0]))
                left_padding = max(0, patch_half_size[1] - x_start)
                right_padding = max(0, x_end - (width - patch_half_size[1]))

                # Apply padding to patches
                patch_volume = np.pad(patch_volume, ((top_padding, bottom_padding), (left_padding, right_padding)),
                                    mode='constant', constant_values=0)
                patch_mask = np.pad(patch_mask, ((top_padding, bottom_padding), (left_padding, right_padding)),
                                    mode='constant', constant_values=0)
                
                # Append the patch and mask to the list
                volume_patch_list.append(patch_volume)
                mask_patch_list.append(patch_mask)

            # Convert lists to numpy arrays with [z, y, x] dimensions
            patch_volume = np.stack(volume_patch_list)
            patch_mask = np.stack(mask_patch_list)

            logging.debug(f"Patch shape: {patch_volume.shape}, Mask shape: {patch_mask.shape}")
            logging.debug(f"Patch height: {len(patch_volume)}, Patch width: {len(patch_volume[0])}")

            logging.debug(f"Patch dimensions: z={patch_volume.shape[0]}, y={patch_volume.shape[1]}, x={patch_volume.shape[2]}")


            patches.append((patch_volume, patch_mask))
        
        return patches

    def __getitem__(self, index):
        """
        Fetches data for a single slice based on index.

        Returns:
            - 2D slice (image, mask) for segmentation tasks.
        """
        # Get patient ID and corresponding slice index
        patient_id = self.slice_index_to_patient_id_list[index]
        first_slice_index = self.patient_id_to_first_index_dict[patient_id]
        dict_slice_index = index - first_slice_index
        slice_index_to_load = self.slices_to_load[patient_id][dict_slice_index]

        # Define the file path to the slice
        file_path = self.path_to_processed_dataset.joinpath(
            f"{patient_id}/slice_{slice_index_to_load}.npy"
        )

        # Load the original image (full CT slice)
        reconstruction_tensor = self.get_reconstruction_tensor(file_path)

        # Apply image transformation if defined
        if self.image_transform is not None:
            reconstruction_tensor = self.image_transform(reconstruction_tensor)

        # Apply lung-only segmentation if specified
        if self.params.task == "segmentation" and self.params.lung_only:
            reconstruction_tensor = self.generate_lung_only_image(reconstruction_tensor)
            reconstruction_tensor = self.normalize_lungs(
                reconstruction_tensor.squeeze(0).cpu().numpy(),
                method=self.params.normalize_lungs
            )
            reconstruction_tensor = torch.from_numpy(reconstruction_tensor).unsqueeze(0)

        # Handle segmentation task (return image and mask)
        if self.params.task in ["segmentation"]:
            mask_tensor = self.get_mask_tensor(patient_id, slice_index_to_load)
            return reconstruction_tensor, mask_tensor

        raise NotImplementedError(f"Task '{self.params.task}' is not implemented.")

    def get_patient_volume(self, patient_id: str):
        """
        Fetches the full volume and corresponding masks for a given patient.
        If the slice thickness is missing, it uses the slice thickness of a neighboring patient.
        """
        # Get slice indices and initial slice thickness
        slice_indices = self.slices_to_load.get(patient_id, [])
        slice_thickness = self.sampling_thickness.get(patient_id)

        # Handle missing slice thickness
        if slice_thickness is None:
            # Attempt to find the thickness of neighboring patients
            patient_index = self.patient_list_to_load.index(patient_id)
            found_thickness = False
            for offset in range(1, len(self.patient_list_to_load)):
                # Look backward
                if patient_index - offset >= 0:
                    neighbor_id = self.patient_list_to_load[patient_index - offset]
                    slice_thickness = self.sampling_thickness.get(neighbor_id)
                    if slice_thickness is not None:
                        found_thickness = True
                        break

                # Look forward
                if patient_index + offset < len(self.patient_list_to_load):
                    neighbor_id = self.patient_list_to_load[patient_index + offset]
                    slice_thickness = self.sampling_thickness.get(neighbor_id)
                    if slice_thickness is not None:
                        found_thickness = True
                        break

            if not found_thickness:
                # Log and return empty tensors if no neighbor has a valid thickness
                with open("patients_missing_slices_or_thickness.txt", "a") as log_file:
                    log_file.write(f"Slice thickness not found for patient {patient_id} and its neighbors.\n")
                return (
                    torch.zeros((1, 1, 512, 512), dtype=torch.float32),
                    torch.zeros((2, 1, 512, 512), dtype=torch.float32),
                )

        # Log the usage of neighboring slice thickness
        if patient_id not in self.sampling_thickness or self.sampling_thickness[patient_id] is None:
            with open("patients_missing_slices_or_thickness.txt", "a") as log_file:
                log_file.write(f"Patient {patient_id} uses slice thickness from neighbor.\n")

        slices = []
        masks = []

        # Load all slices and masks for the patient
        for slice_index in slice_indices:
            file_path = self.path_to_processed_dataset.joinpath(
                f"{patient_id}/slice_{slice_index}.npy"
            )

            if not file_path.exists():
                logging.debug(f"File not found: {file_path}")
                continue

            slice_image = self.get_reconstruction_tensor(file_path)
            if slice_image is None or slice_image.numel() == 0:
                logging.debug(f"Invalid slice image for file: {file_path}")
                continue

            mask = self.get_mask_tensor(patient_id, slice_index)
            if mask is None or mask.numel() == 0:
                logging.debug(f"Invalid mask for slice {slice_index} of patient {patient_id}")
                continue

            # Apply image transformation if defined
            if self.image_transform is not None:
                slice_image = self.image_transform(slice_image)

            if self.params.task == "segmentation" and self.params.lung_only:
                slice_image = self.generate_lung_only_image(slice_image)

            slices.append(slice_image.squeeze(0).cpu().numpy())  # Convert to numpy
            masks.append(mask.cpu().numpy())

        logging.debug(f"Loaded volume for patient {patient_id}. Shape: {len(slices)} slices, {len(masks)} masks. Slice thickness: {slice_thickness}")
        # Check if slices were loaded
        if not slices:
            logging.debug(f"No valid slices found for patient {patient_id}. Returning empty tensors.")
            return torch.zeros((1, 1, 512, 512), dtype=torch.float32), torch.zeros((2, 1, 512, 512), dtype=torch.float32)

        # Stack slices to form 3D volume
        volume = np.stack(slices, axis=0)  # Shape: (depth, 512, 512)
        mask_volume = np.stack(masks, axis=0)  # Shape: (depth, 2, 512, 512)
        logging.debug(f"Loaded volume for patient {patient_id}. Shape: {volume.shape} slices, {mask_volume.shape} masks.")
        mask_volume = np.stack(masks, axis=0)  # Shape: (depth, 2, 512, 512)
        tumor_mask = mask_volume[:, 1, :, :]  # Extract the tumor channel
        logging.debug(f"Mask volume shape: {mask_volume.shape}, Tumor mask shape: {tumor_mask.shape}")
        
        logging.debug(f"Min tumor size: {self.params.min_tumor_size}, Max tumor size: {self.params.max_tumor_size}")
        # Apply tumor size filtering
        filtered_tumor_mask = self.filter_tumors_by_size(
            tumor_mask,
            min_size=self.params.min_tumor_size,
            max_size=self.params.max_tumor_size
        )

        logging.debug(f"Filtered tumor mask has {filtered_tumor_mask.sum()} non-zero elements.")

        
        # Update the mask volume with the filtered tumors
        mask_volume[:, 1, :, :] = filtered_tumor_mask

        logging.debug(f"Volume and mask shapes after tumor filtering: {volume.shape}, {mask_volume.shape}")

        # Interpolate to finer z-spacing
        original_spacing = (slice_thickness, 1.0, 1.0)  # Original (z, y, x) spacing
        target_spacing = self.voxel_spacing
        volume = self.interpolate_volume(
                volume=volume,
                original_spacing=original_spacing,
                target_spacing=target_spacing,
                method=self.params.interpolation_method,
                order=self.params.bspline_order,           # Relevant for B-Spline
                sinc_width=self.params.welch_sinc_width    # Relevant for Welch Sinc
            )

        # Interpolate each mask channel separately with nearest-neighbor interpolation
        mask_channels = []
        for channel in range(mask_volume.shape[1]):  # Iterate over channels (background and nodule)
            mask_channel = self.interpolate_volume(
                mask_volume[:, channel, :, :],
                original_spacing,
                target_spacing,
                order=1  # Nearest-neighbor interpolation
            )
            mask_channels.append(mask_channel)

        mask_volume = np.stack(mask_channels, axis=0)  # Shape: (2, depth, 512, 512)
        logging.debug(f"Volume and mask shapes after interpolation: {volume.shape}, {mask_volume.shape}")
        logging.debug(f"Interpolated mask volume for patient {patient_id}. Shape: {mask_volume.shape}")

        # Normalize the volume if lung_only is active
        if self.params.task == "segmentation" and self.params.lung_only:
            volume = self.normalize_lungs(volume, method=self.params.normalize_lungs)

        logging.debug(f"Volume and Mask shapes before calling tumor patch extraction: {volume.shape}, {mask_volume.shape}")
    
        if self.params.tumor_patch_mode:

            logging.debug(f"Tumor patch mode enabled. Extracting tumor-centered patches.")
            
            # Extract tumor-centered patches after filtering
            tumor_volumes = self.extract_tumor_centered_patches(volume, mask_volume, self.params.patch_size)

            # Return one volume per tumor
            patches = [(
                torch.from_numpy(patch_volume).unsqueeze(0),  # Shape: (1, patch_depth, patch_height, patch_width)
                torch.from_numpy(patch_mask)     # Shape: (1, patch_depth, patch_height, patch_width)
            ) for patch_volume, patch_mask in tumor_volumes]
            logging.debug(f"Number of patches: {len(patches)}")
            if len(patches) > 0:
                logging.debug(f"Volume patch tensor shape: {patches[0][0].shape}, Mask patch tensor shape: {patches[0][1].shape}")
            return patches

        else:
            logging.debug(f"Returning full volume for patient {patient_id}. Shape: {volume.shape}")
            # Convert to PyTorch tensors
            volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # Shape: (1, depth, 512, 512)
            mask_tensor = torch.from_numpy(mask_volume)  # Shape: (2, depth, 512, 512)

            logging.debug(f"Volume tensor shape: {volume_tensor.shape}, Mask tensor shape: {mask_tensor.shape}")

            return volume_tensor, mask_tensor

        
    def preprocess_pipeline_setup(self):
        """
        Sets up parameters and configurations for the lung segmentation pipeline.
        Initializes kernel sizes, iteration counts, and other parameters for 
        the morphological operations.
        """
        # Define kernel size and iterations for dilation and erosion operations.
        # These can be adjusted based on the size and resolution of CT slices.
        
        # Number of iterations for dilation and erosion
        self.dilation_iterations = 2
        self.erosion_iterations = 2
        
        # Store thresholding method, if it might be changed later
        self.thresholding_method = "otsu"  # "otsu" is default; other methods can be added

    def generate_lung_only_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies the lung segmentation pipeline to return a lung-only representation of the input slice.
        """
        # Convert to numpy for processing if not already
        slice_np = image.squeeze().cpu().numpy() if isinstance(image, torch.Tensor) else image

        # Step 1: Threshold Segmentation (Otsu’s Method)
        threshold = threshold_otsu(slice_np)
        binary_mask = slice_np > threshold

        # Step 2: Remove the Background and Isolate the Body and Lung Regions Together
        # Label the connected components in the original binary mask
        # The largest connected component along the edges should be the background

        # Invert the binary mask so the background is white, body and lungs are black
        inverted_mask = np.invert(binary_mask)

        # Label connected components in the inverted mask
        labeled_mask, num_labels = label(inverted_mask)

        # Identify the largest component, which should be the outer background
        if num_labels > 1:
            largest_blob_label = np.argmax([np.sum(labeled_mask == label) for label in range(1, num_labels + 1)]) + 1
            background_mask = (labeled_mask == largest_blob_label)

        else:
            background_mask = np.zeros_like(binary_mask)

        # Combine background with the non-lung regions (body)
        # Invert again to set the background and body to white, keeping only lung regions black
        combined_mask = np.invert(background_mask | binary_mask)

        # Step 3: Fill Holes in the Lung Mask
        filled_mask = binary_fill_holes(combined_mask)

        # Step 4: Dilation and Erosion
        # Dilation followed by erosion to ensure smooth lung boundaries
        dilated_mask = binary_dilation(filled_mask, iterations=2)
        final_mask = binary_erosion(dilated_mask, iterations=2)

        # Apply mask to original image
        lung_only_image = slice_np * final_mask

        # Convert back to torch.Tensor for consistency in dataloader output
        return torch.from_numpy(lung_only_image).unsqueeze(0)


class VolumeWindowDataloader(Dataset):
    def __init__(self, dataset, window_depth, pad_value=0, stride=1, nodule_only=True):
        """
        A dynamic dataloader for processing volumes as they are accessed.

        Parameters:
            - dataset (LIDC_IDRI): The dataset instance, which determines mode and patients.
            - window_depth (int): Fixed depth of each window.
            - pad_value (float): Value to pad volumes smaller than the window depth.
            - stride (int): The step size for the sliding window.
            - nodule_only (bool): If True, only include slices with nodules.
        """
        self.dataset = dataset
        self.window_depth = window_depth
        self.pad_value = pad_value
        self.stride = stride
        self.nodule_only = nodule_only

        self.mode = dataset.params.mode
        if self.mode == "train":
            self.patient_ids = dataset.training_patients_list
        elif self.mode == "validation":
            self.patient_ids = dataset.validation_patients_list
        elif self.mode == "test":
            self.patient_ids = dataset.testing_patients_list
        else:
            raise ValueError(f"Unsupported dataset mode: {self.mode}")

    def __len__(self):
        """Returns the number of patients in the dataset."""
        return len(self.patient_ids)

    def __getitem__(self, index):
        """
        Dynamically processes a single volume from the dataset and yields its sliding windows.
        
        Parameters:
            - index: Index of the patient in the dataset.

        Returns:
            - A tuple (windows, mask_windows) for sliding windows.
        """
        # Fetch patient ID and corresponding volume and mask
        patient_id = self.patient_list_to_load[index]
        volume, mask = self.dataset.get_patient_volume(patient_id)

        # If no valid windows exist for the patient, skip them
        if volume.shape[1] == 1 and mask.shape[1] == 2:
            # print(f"Skipping patient {patient_id} as there are no valid windows with nodules.")
            return None  # Skip empty windows

        depth = volume.shape[1]

        # Pad the volume and mask if the depth is smaller than the window depth
        if depth < self.window_depth:
            pad_size = self.window_depth - depth
            volume = torch.nn.functional.pad(volume, (0, 0, 0, 0, pad_size, 0), value=self.pad_value)
            mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, pad_size, 0), value=self.pad_value)
            depth = self.window_depth  # Update depth after padding

        # Create sliding windows
        windows = []
        mask_windows = []
        for start_idx in range(0, depth - self.window_depth + 1, self.stride):
            end_idx = start_idx + self.window_depth
            if end_idx > depth:  # Safety check
                break
            window = volume[:, start_idx:end_idx, :, :]
            mask_window = mask[:, start_idx:end_idx, :, :]

            if self.nodule_only:
                # Check if the mask window contains any nodules
                if mask_window[1].sum() == 0:  # No nodules found in this window
                    continue  # Skip this window if no nodules

            windows.append(window)
            mask_windows.append(mask_window)

        # If no valid windows, return None
        if len(windows) == 0:
            # print(f"Skipping patient {patient_id} as no valid windows were found with nodules.")
            return None  # Skip empty windows

        # Ensure the last window includes the final slices of the volume
        if len(windows) == 0 or windows[-1].shape[1] < self.window_depth:
            start_idx = max(0, depth - self.window_depth)
            window = volume[:, start_idx:, :, :]
            mask_window = mask[:, start_idx:, :, :]

            # Pad the last window if needed
            if window.shape[1] < self.window_depth:
                pad_size = self.window_depth - window.shape[1]
                window = torch.nn.functional.pad(window, (0, 0, 0, 0, 0, pad_size), value=self.pad_value)
                mask_window = torch.nn.functional.pad(mask_window, (0, 0, 0, 0, 0, pad_size), value=self.pad_value)
            
            if self.nodule_only:
                # Check if the mask window contains any nodules
                if mask_window[1].sum() != 0:  # Check if the nodule is present
                    windows.append(window)
                    mask_windows.append(mask_window)
            else:
                windows.append(window)
                mask_windows.append(mask_window)

        # Stack windows into tensors
        windows = torch.stack(windows)
        mask_windows = torch.stack(mask_windows)

        return windows, mask_windows





