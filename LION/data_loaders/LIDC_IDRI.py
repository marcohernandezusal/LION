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

from LION.utils.paths import LIDC_IDRI_PROCESSED_DATASET_PATH
import LION.CTtools.ct_utils as ct
from LION.utils.parameter import LIONParameter


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
            return len(self.patient_ids)  # Number of patients
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

    def interpolate_volume(self, volume, original_spacing, target_spacing, order=1):
        """
        Resample a 3D volume to the target spacing.

        Parameters:
            - volume (np.ndarray): Input 3D volume (z, y, x).
            - original_spacing (tuple): Original voxel spacing (z, y, x).
            - target_spacing (tuple): Desired voxel spacing (z, y, x).
            - order (int): Interpolation order. Default is 1 (linear). Use 0 for nearest-neighbor (binary masks).

        Returns:
            - Resampled 3D volume (np.ndarray).
        """
        assert len(original_spacing) == volume.ndim, (
            f"Original spacing {original_spacing} must match volume dimensions {volume.ndim}."
        )
        assert len(target_spacing) == volume.ndim, (
            f"Target spacing {target_spacing} must match volume dimensions {volume.ndim}."
        )
        
        resize_factors = [original / target for original, target in zip(original_spacing, target_spacing)]

        return zoom(volume, resize_factors, order=order)

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

        Parameters:
            - patient_id (str): The ID of the patient.

        Returns:
            - volume_tensor (torch.Tensor): The 3D volume (1, depth, 512, 512).
            - mask_tensor (torch.Tensor): The 3D mask (2, depth, 512, 512).
        """
        # print(f"Fetching volume for patient {patient_id}")

        # Get slice indices and thickness for interpolation
        slice_indices = self.slices_to_load.get(patient_id, [])
        slice_thickness = self.sampling_thickness.get(patient_id)


        # print(f"Found {len(slice_indices)} slices for patient {patient_id}")

        # Open a file to log all the patients for which no slices or slice thickness is not found
        log_file_path = "patients_missing_slices_or_thickness.txt"

        # Open the file in append mode to ensure we don't overwrite previous logs
        with open(log_file_path, "a") as log_file:

            if not slice_indices:
                log_file.write(f"No slice indices found for patient {patient_id}.\n")
                # print(f"No slice indices found for patient {patient_id}.")
                return torch.zeros((1, 1, 512, 512), dtype=torch.float32), torch.zeros((2, 1, 512, 512), dtype=torch.float32)

            if slice_thickness is None:
                log_file.write(f"Slice thickness not found for patient {patient_id}.\n")
                # print(f"Slice thickness not found for patient {patient_id}.")
                return torch.zeros((1, 1, 512, 512), dtype=torch.float32), torch.zeros((2, 1, 512, 512), dtype=torch.float32)

        slices = []
        masks = []

        # Load all slices and masks for the patient
        for slice_index in slice_indices:
            file_path = self.path_to_processed_dataset.joinpath(
                f"{patient_id}/slice_{slice_index}.npy"
            )

            if not file_path.exists():
                print(f"File not found: {file_path}")
                continue

            slice_image = self.get_reconstruction_tensor(file_path)
            if slice_image is None or slice_image.numel() == 0:
                print(f"Invalid slice image for file: {file_path}")
                continue

            mask = self.get_mask_tensor(patient_id, slice_index)
            if mask is None or mask.numel() == 0:
                print(f"Invalid mask for slice {slice_index} of patient {patient_id}")
                continue

            # Apply image transformation if defined
            if self.image_transform is not None:
                slice_image = self.image_transform(slice_image)

            if self.params.task == "segmentation" and self.params.lung_only:
                slice_image = self.generate_lung_only_image(slice_image)

            slices.append(slice_image.squeeze(0).cpu().numpy())  # Convert to numpy
            masks.append(mask.cpu().numpy())

        # Check if slices were loaded
        if not slices:
            print(f"No valid slices found for patient {patient_id}. Returning empty tensors.")
            return torch.zeros((1, 1, 512, 512), dtype=torch.float32), torch.zeros((2, 1, 512, 512), dtype=torch.float32)

        # Stack slices to form 3D volume
        volume = np.stack(slices, axis=0)  # Shape: (depth, 512, 512)
        mask_volume = np.stack(masks, axis=0)  # Shape: (depth, 2, 512, 512)

        # Interpolate to finer z-spacing
        original_spacing = (slice_thickness, 1.0, 1.0)  # Original (z, y, x) spacing
        target_spacing = self.voxel_spacing
        volume = self.interpolate_volume(volume, original_spacing, target_spacing, order=1)

        # Interpolate each mask channel separately with nearest-neighbor interpolation
        mask_channels = []
        for channel in range(mask_volume.shape[1]):  # Iterate over channels (background and nodule)
            mask_channel = self.interpolate_volume(
                mask_volume[:, channel, :, :],
                original_spacing,
                target_spacing,
                order=0  # Nearest-neighbor interpolation
            )
            mask_channels.append(mask_channel)

        mask_volume = np.stack(mask_channels, axis=0)  # Shape: (2, depth, 512, 512)

        # Normalize the volume if lung_only is active
        if self.params.task == "segmentation" and self.params.lung_only:
            volume = self.normalize_lungs(volume, method=self.params.normalize_lungs)

        # Convert to PyTorch tensors
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # Shape: (1, depth, 512, 512)
        mask_tensor = torch.from_numpy(mask_volume)  # Shape: (2, depth, 512, 512)

        # print(f"Resampled volume shape: {volume_tensor.shape}, Mask shape: {mask_tensor.shape}")
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
    def __init__(self, dataset, window_depth, pad_value=0, stride=1, nodule_only=False):
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
        patient_id = self.patient_ids[index]
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





