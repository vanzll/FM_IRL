# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Import DDPM modules conditionally
try:
    from .ddpm_customize import MLPDiffusionCustomize
except ImportError as e:
    print(f"Warning: MLPDiffusionCustomize import failed: {e}")
    MLPDiffusionCustomize = None

try:
    from .ddpm_condition import MLPConditionDiffusion
except ImportError as e:
    print(f"Warning: MLPConditionDiffusion import failed: {e}")
    MLPConditionDiffusion = None
