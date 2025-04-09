# create alignment
from typing import Dict, Optional, List, Any

import numpy as np
from numpy import ndarray, dtype, floating

from dfi_data_comparison import align_dfi_to_msa, svg_analysis_of_dfi, svg_analysis_of_dfi_3D
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import pairwise_distances

import subprocess



def run_clustalo(in_fasta):
    clustalo_path = "clustalo"
    output_msa_path = in_fasta.replace('.fasta', '_msa.fasta')

    try:
        cmd = [clustalo_path, "-i", in_fasta, "-o", output_msa_path, "--force"]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during sequence alignment: {e}")
    except FileNotFoundError:
        print(f"Clustal Omega executable not found at {clustalo_path}. Please check your installation.")
    return output_msa_path


# Specify your values here
def align_md_profiles_to_MSA(
        md_data: Dict[str, np.ndarray],
        fasta_path: Optional[str],
        msa_path: Optional[str],
        svg_R: Optional[int],
        remove_msa_gap_columns: bool = True,
        calculate_correlations: bool = False,
        calculate_distance: bool = False,
        required_order: List[str] = None,
) -> tuple[ndarray[dtype[floating[Any]]], list[str]]:
    assert calculate_correlations != calculate_distance, "Error: choose either calculate_correlations xor calculate_distance"
    assert fasta_path or msa_path, "Error: either fasta_path or msa_path must be set"

    if msa_path:
        print(f"   Using MSA specified in the path {msa_path}")
    else:
        print(f"   Aligning sequences specified in the path {fasta_path}")
        msa_path = run_clustalo(fasta_path)
        print(f"   Sequences aligned to MSA: {msa_path}")

    data_shape_original = len(list(md_data.values())[0].shape)
    print(f"   Shape of original data: {data_shape_original}")
    aligned_statistic_matrix, msa_order_names = align_dfi_to_msa(md_data, msa_path, remove_msa_gap_columns)
    print(aligned_statistic_matrix.shape)
    # print(aligned_statistic_matrix)

    if svg_R is not None:
        print(f"   Calculation of SVG with R={svg_R}")
        if data_shape_original == 2:
            X_r = svg_analysis_of_dfi_3D(aligned_statistic_matrix, r=svg_R, return_vector=True)
            X_r = X_r.reshape((len(aligned_statistic_matrix), -1))
        elif data_shape_original == 1:
            X_r = svg_analysis_of_dfi(aligned_statistic_matrix, r=svg_R)
        else:
            print(f"   SVG with R={svg_R} not implemented for dimensionality of data {data_shape_original}")
            exit()
    else:
        X_r = aligned_statistic_matrix
        if data_shape_original == 2:
            X_r = X_r.reshape((len(aligned_statistic_matrix), -1))

    if required_order:
        ordered_X_r = np.zeros(X_r.shape)
        msa_rename_order = []
        for order_i, w in enumerate(required_order):
            try:
                msa_index = msa_order_names.index(w)
                msa_rename_order.append(msa_order_names[msa_index])
                ordered_X_r[order_i] = X_r[msa_index]
            except ValueError:
                print(f"MSA {msa_order_names[order_i]} does not match {w}")
                exit()
    else:
        ordered_X_r = X_r
        msa_rename_order = msa_order_names

    if calculate_correlations:
        final_matrix = np.corrcoef(ordered_X_r)
    elif calculate_distance:
        # Compute pairwise Euclidean distances between rows of the reduced matrix X_r
        # distance_matrix = squareform(pdist(ordered_X_r, metric='euclidean'))  # Pairwise distances
        final_matrix = pairwise_distances(ordered_X_r, metric='euclidean')
        # final_matrix = distance_matrix
        # final_matrix = 1 - (distance_matrix / distance_matrix.max())
    else:
        print("   Please choose calculate_correlations xor calculate_distance")
        exit()

    return final_matrix, msa_rename_order