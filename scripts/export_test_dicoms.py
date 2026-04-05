#!/usr/bin/env python3
"""
Export test-set RSS reconstructions as DICOM files for viewing.

Directory structure:
  <output_dir>/
    exam_001/
      pd/   *.dcm   (one per slice)
      pdfs/ *.dcm   (one per slice)
    exam_002/
      ...

Each exam corresponds to one row in the paired CSV (same patient, same session).
"""

import argparse
import os
import sys
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def make_dicom(pixel_array, series_uid, study_uid, patient_id,
               series_desc, instance_number, series_number):
    """Create a minimal DICOM dataset from a 2D numpy array."""
    # Normalize to 16-bit unsigned
    arr = pixel_array.astype(np.float64)
    if arr.max() > 0:
        arr = arr / arr.max() * 65535.0
    arr = arr.astype(np.uint16)

    ds = Dataset()
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = generate_uid()

    ds.PatientID = patient_id
    ds.PatientName = patient_id
    ds.StudyDescription = "fastMRI knee"
    ds.SeriesDescription = series_desc
    ds.Modality = "MR"
    ds.Manufacturer = "fastMRI"
    ds.StudyDate = datetime.now().strftime("%Y%m%d")
    ds.SeriesNumber = series_number
    ds.InstanceNumber = instance_number
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, float(instance_number)]
    ds.SliceLocation = float(instance_number)
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 3.0
    ds.SpacingBetweenSlices = 3.0

    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # unsigned
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = arr.tobytes()
    ds.NumberOfFrames = 1
    ds.WindowCenter = int(arr.max() // 2)
    ds.WindowWidth = int(arr.max())

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    return ds


def export_exam(data_dir, output_dir, exam_label, pd_fname, pdfs_fname,
                study_uid, patient_id):
    """Export one exam (PD + PDFS) to DICOM."""
    for contrast, fname, series_number in [("pd", pd_fname, 1), ("pdfs", pdfs_fname, 2)]:
        if pd.isna(fname):
            continue
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found, skipping")
            continue

        series_uid = generate_uid()
        series_desc = f"RSS {contrast.upper()} - {fname}"
        out_series = os.path.join(output_dir, exam_label, contrast)
        os.makedirs(out_series, exist_ok=True)

        with h5py.File(fpath, "r") as f:
            rss_vol = f["reconstruction_rss"][:]  # (slices, 320, 320)

        for sl in range(rss_vol.shape[0]):
            ds = make_dicom(
                rss_vol[sl],
                series_uid=series_uid,
                study_uid=study_uid,
                patient_id=patient_id,
                series_desc=series_desc,
                instance_number=sl + 1,
                series_number=series_number,
            )
            dcm_path = os.path.join(out_series, f"slice_{sl:03d}.dcm")
            ds.save_as(dcm_path, write_like_original=False)

        print(f"  {contrast.upper():4s}: {rss_vol.shape[0]} slices -> {out_series}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="/gpfs/scratch/johnsp23/DLrecon_lab1/data/knee")
    parser.add_argument("--split_csv", default="/gpfs/scratch/johnsp23/DLrecon_lab1/data/fastMRI_paired_knee.csv")
    parser.add_argument("--output_dir", default="/gpfs/scratch/johnsp23/DLrecon_lab1/dicoms_test")
    args = parser.parse_args()

    df = pd.read_csv(args.split_csv)

    # Each row where either contrast is in the test split
    test_rows = df[(df["pd_split"] == "test") | (df["pdfs_split"] == "test")]
    print(f"Exporting {len(test_rows)} exams to {args.output_dir}\n")

    for i, (_, row) in enumerate(test_rows.iterrows()):
        exam_label = f"exam_{i+1:03d}"
        patient_id = exam_label
        study_uid = generate_uid()

        print(f"[{i+1}/{len(test_rows)}] {exam_label}  "
              f"pd={row['pd']}  pdfs={row['pdfs']}")
        export_exam(
            args.data_dir, args.output_dir, exam_label,
            row["pd"], row["pdfs"], study_uid, patient_id,
        )

    print(f"\nDone. DICOMs in: {args.output_dir}")


if __name__ == "__main__":
    main()
