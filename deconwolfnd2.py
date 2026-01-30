#!/usr/bin/env python3
import os
import re
import argparse
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import tifffile

# ----------------------------
# MERlin-style parsing helpers
# ----------------------------
def _parse_list(inputString: str, dtype=float):
    if isinstance(inputString, (list, np.ndarray)):
        return np.asarray(inputString, dtype=dtype)
    s = str(inputString)
    if ',' in s:
        return np.fromstring(s.strip('[] '), dtype=dtype, sep=',')
    else:
        return np.fromstring(s.strip('[] '), dtype=dtype, sep=' ')

def _parse_int_list(inputString: str):
    return _parse_list(inputString, dtype=int)

def _parse_conditional_int_list(inputString: str):
    arr = _parse_list(inputString, dtype=int)
    if len(arr) == 1:
        return int(arr[0])
    return arr

# ----------------------------
# ND2 reader (your MERlin style)
# ----------------------------
from nd2 import ND2File

class ND2Reader:
    """
    MERlin-style ND2 reader that provides:
      - number_frames
      - frame_2_channel_index: {frame_number: (channel_index, z_index)}
      - load_frame(frame_number) -> 2D image
    """
    def __init__(self, filename: str, verbose: bool = False):
        self.filename = filename
        self.verbose = verbose
        self._parse_nd2()

    def _parse_nd2(self):
        with ND2File(self.filename) as ndfile:
            metadata = ndfile.metadata
            self.voxel_sizes = ndfile.voxel_size()

            _ws, _hs, _zs = [], [], []
            for channel in metadata.channels:
                _w, _h, _z = channel.volume.voxelCount
                _ws.append(_w)
                _hs.append(_h)
                _zs.append(_z)

            if np.unique(_ws).size != 1:
                raise ValueError("Different frame widths across channels are not supported.")
            if np.unique(_hs).size != 1:
                raise ValueError("Different frame heights across channels are not supported.")

            self.image_width = int(np.unique(_ws)[0])
            self.image_height = int(np.unique(_hs)[0])
            self.channels_names = [c.channel.name for c in metadata.channels]
            self.zs = list(map(int, _zs))

        # Global frame index over (channel, z) blocks
        self.number_frames = int(np.sum(self.zs))
        self.frame_2_channel_index: Dict[int, Tuple[int, int]] = {}
        k = 0
        for ci, nz in enumerate(self.zs):
            for zj in range(nz):
                self.frame_2_channel_index[k] = (ci, zj)
                k += 1

    def load_frame(self, frame_number: int) -> np.ndarray:
        if frame_number < 0 or frame_number >= self.number_frames:
            raise IndexError(f"frame_number={frame_number} out of bounds [0, {self.number_frames-1}]")

        with ND2File(self.filename) as ndfile:
            ci, zj = self.frame_2_channel_index[frame_number]
            # Your original assumption:
            # ndfile.asarray(0) returns something indexable by [z][channel]
            # If your ND2 differs, this is the one line you may need to adjust.
            arr = ndfile.asarray(0)
            img = arr[zj][ci]
        return img

# ----------------------------
# DeconWolf wrapper (file-based CLI)
# ----------------------------
def run_deconwolf(
    img: np.ndarray,
    channel_key: str,
    dw_path: str,
    ref_path: str,
    zstep: int = 1200,
    tile_size: int = 800,
    gpu: bool = False,
    n_iter: int = 100,
    scale: float = 1.0,
    tile: bool = True,
    overwrite: bool = True,
) -> np.ndarray:
    """
    img can be 2D (y,x) or 3D (z,y,x) depending on how your dw build is configured.
    This wrapper simply writes TIFF, calls dw, reads TIFF back.
    """

    channel_2_psf = {
        "748": "Alexa750",
        "637": "Alexa647",
        "545": "Atto565",
        "477": "beads",
        "488": "Alexa488",
        "405": "DAPI",
    }

    if not os.path.exists(dw_path):
        raise FileNotFoundError(f"DeconWolf executable not found: {dw_path}")
    if channel_key not in channel_2_psf:
        raise KeyError(f"channel_key={channel_key} not in channel_2_psf mapping")

    # Find PSF
    psf_tag = channel_2_psf[channel_key]
    matched = [
        os.path.join(ref_path, f)
        for f in os.listdir(ref_path)
        if (psf_tag in f) and (str(zstep) in f) and ("psf" in f) and f.lower().endswith(".tif")
    ]
    if len(matched) == 0:
        raise FileNotFoundError(f"No PSF found for channel={channel_key}, zstep={zstep} under {ref_path}")
    if len(matched) > 1:
        raise FileNotFoundError(f"Multiple PSFs found for channel={channel_key}, zstep={zstep}: {matched}")
    psf_path = matched[0]

    with tempfile.TemporaryDirectory() as tmp:
        in_tif = os.path.join(tmp, "img.tif")
        out_tif = os.path.join(tmp, "decon.tif")
        tifffile.imwrite(in_tif, img)

        gpu_flag = " --gpu" if gpu else ""
        tile_flag = f" --tilesize {tile_size}" if tile else ""
        scale_flag = f" --scaling {scale}" if scale else ""
        overwrite_flag = " --overwrite" if overwrite else ""

        cmd = (
            f"{dw_path} --out {out_tif} --iter {n_iter}"
            f"{scale_flag}{tile_flag}{gpu_flag}{overwrite_flag} --verbose 1 "
            f"{in_tif} {psf_path}"
        )
        subprocess.run(cmd, check=True, shell=True)
        decon = tifffile.imread(out_tif)

    return decon

# ----------------------------
# DataOrganization + fileMap from imageRegExp (no merlin Dataset required)
# ----------------------------
@dataclass(frozen=True)
class FileKey:
    imageType: str
    fov: int
    imagingRound: int

def list_files_recursive(root: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(exts):
                out.append(os.path.join(r, fn))
    return out

def build_filemap_from_dataorganization(
    raw_data_path: str,
    dataorg_df: pd.DataFrame,
    allowed_exts: Tuple[str, ...] = (".nd2",),
) -> pd.DataFrame:
    """
    Scans raw_data_path for files, matches against each unique (imageType,imageRegExp),
    and returns MERlin-like fileMap with columns: imageType,fov,imagingRound,imagePath
    where imagePath is relative to raw_data_path.
    """
    file_names = list_files_recursive(raw_data_path, allowed_exts)
    if len(file_names) == 0:
        raise FileNotFoundError(f"No files with extensions {allowed_exts} under {raw_data_path}")

    unique_entries = dataorg_df.drop_duplicates(subset=["imageType", "imageRegExp"], keep="first")
    file_rows = []

    # match against path relative to raw_data_path
    for _, row in unique_entries.iterrows():
        current_type = str(row["imageType"])
        match_re = re.compile(str(row["imageRegExp"]))

        any_match = False
        for abs_path in file_names:
            rel = os.path.relpath(abs_path, raw_data_path)
            m = match_re.match(rel)
            if m is None:
                continue
            gd = m.groupdict()
            # Require groupdict keys
            if "imageType" not in gd or "fov" not in gd:
                raise ValueError(
                    f"imageRegExp must define (?P<imageType>...) and (?P<fov>...) named groups.\n"
                    f"imageType={current_type}, imageRegExp={row['imageRegExp']}"
                )
            if gd["imageType"] != current_type:
                continue

            if "imagingRound" not in gd or gd["imagingRound"] in (None, ""):
                imaging_round = -1
            else:
                imaging_round = int(gd["imagingRound"])

            file_rows.append(
                {
                    "imageType": gd["imageType"],
                    "fov": int(gd["fov"]),
                    "imagingRound": int(imaging_round),
                    "imagePath": rel,
                }
            )
            any_match = True

        if not any_match:
            raise FileNotFoundError(
                f"Unable to find files for imageType={current_type} with imageRegExp={row['imageRegExp']}\n"
                f"raw_data_path={raw_data_path}"
            )

    fm = pd.DataFrame(file_rows).drop_duplicates()
    fm[["fov", "imagingRound"]] = fm[["fov", "imagingRound"]].astype(int)
    return fm

def load_dataorganization_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        converters={
            "frame": _parse_int_list,
            "zPos": _parse_list,
            "fiducialFrame": _parse_conditional_int_list,
        },
    )
    # Keep string columns as strings
    for col in ["readoutName", "channelName", "imageType", "imageRegExp", "fiducialImageType", "fiducialRegExp"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

# ----------------------------
# ND2 -> MERlin-style TIFF export with optional decon
# ----------------------------
def choose_frames_to_decon(
    decon_mode: str,
    dataorg_df: pd.DataFrame,
    imageType: str,
    imagingRound: int,
) -> Set[int]:
    """
    Returns a set of global frame indices to deconvolve.
    - "all": deconvolve every frame in the ND2
    - "readouts": deconvolve only frames referenced by dataorganization.frame for rows matching (imageType, imagingRound)
    - "none": empty set
    """
    mode = decon_mode.lower()
    if mode == "none":
        return set()
    if mode == "all":
        # handled by caller once ND2 frame count is known
        return {"__ALL__"}  # sentinel
    if mode == "readouts":
        sub = dataorg_df[(dataorg_df["imageType"] == imageType) & (dataorg_df["imagingRound"] == imagingRound)]
        frames = set()
        for v in sub["frame"].values:
            arr = np.asarray(v).reshape(-1)
            frames.update([int(x) for x in arr])
        return frames
    raise ValueError(f"Unknown decon_mode={decon_mode}. Use one of: none, all, readouts")

def export_merlin_tifs_from_nd2_flat(
    raw_data_path: str,
    output_path: str,
    dataorg_df: pd.DataFrame,
    filemap_df: pd.DataFrame,
    dw_path: str,
    ref_path: str,
    fov_filter: Optional[int] = None,
    decon_mode: str = "readouts",
    # decon params
    gpu: bool = False,
    tile: bool = True,
    tile_size: int = 800,
    zstep: int = 1200,
    n_iter: int = 100,
    scale: float = 1.0,
    overwrite: bool = False,
    # mapping from ND2 channel name -> deconwolf channel_key ("637","748",...)
    # If you know your ND2 channel naming, set this accordingly.
    nd2_channelname_to_key: Optional[Dict[str, str]] = None,
):
    os.makedirs(output_path, exist_ok=True)

    # Default mapping is heuristic: look for wavelength substrings in channel name
    if nd2_channelname_to_key is None:
        nd2_channelname_to_key = {
            "750": "748",
            "748": "748",
            "647": "637",
            "637": "637",
            "565": "545",
            "561": "545",
            "488": "488",
            "405": "405",
            "bead": "477",
            "477": "477",
        }

    # iterate all (imageType,fov,imagingRound) entries
    fm = filemap_df.copy()
    if fov_filter is not None:
        fm = fm[fm["fov"] == int(fov_filter)]

    for _, r in fm.sort_values(["imageType", "fov", "imagingRound"]).iterrows():
        imageType = str(r["imageType"])
        fov = int(r["fov"])
        imagingRound = int(r["imagingRound"])
        rel_path = str(r["imagePath"])
        abs_path = os.path.join(raw_data_path, rel_path)

        out_name = f"{imageType}_{fov}_{imagingRound}.tif"
        out_path = os.path.join(output_path, out_name)
        if os.path.exists(out_path) and not overwrite:
            print(f"Skip existing: {out_path}")
            continue

        print(f"\nProcessing ND2: {abs_path}")
        reader = ND2Reader(abs_path)

        # Decide frames to deconvolve
        frames_to_decon = choose_frames_to_decon(decon_mode, dataorg_df, imageType, imagingRound)
        if "__ALL__" in frames_to_decon:
            frames_to_decon = set(range(reader.number_frames))

        # Load all frames (y,x) into stack (frames,y,x) to preserve MERlin frame indexing
        stack = np.empty((reader.number_frames, reader.image_height, reader.image_width), dtype=np.uint16)

        # Cache mapping: channel_index -> channel_key for deconwolf
        # Based on channel name strings from ND2 metadata
        channel_index_to_key: Dict[int, Optional[str]] = {}
        for ci, cname in enumerate(reader.channels_names):
            cname_l = cname.lower()
            key = None
            for token, mapped in nd2_channelname_to_key.items():
                if token in cname_l:
                    key = mapped
                    break
            channel_index_to_key[ci] = key  # may be None

        for frame in range(reader.number_frames):
            img = reader.load_frame(frame)
            if img.dtype != np.uint16:
                # keep your pipeline consistent; adjust if you want float
                img = img.astype(np.uint16, copy=False)

            if frame in frames_to_decon:
                ci, _zj = reader.frame_2_channel_index[frame]
                ch_key = channel_index_to_key.get(ci, None)
                if ch_key is None:
                    # No PSF mapping for this channel name; just leave as-is
                    stack[frame] = img
                else:
                    decon = run_deconwolf(
                        img=img,
                        channel_key=ch_key,
                        dw_path=dw_path,
                        ref_path=ref_path,
                        zstep=zstep,
                        tile_size=tile_size,
                        gpu=gpu,
                        n_iter=n_iter,
                        scale=scale,
                        tile=tile,
                        overwrite=True,
                    )
                    # Decon output might be float/other; cast back
                    stack[frame] = np.asarray(decon, dtype=np.uint16)
            else:
                stack[frame] = img

        # Write ImageJ-compatible TIFF with frames as pages.
        # MERlin frame indices correspond to page index.
        print(f"Writing: {out_path}  (frames={stack.shape[0]})")
        tifffile.imwrite(out_path, stack, imagej=True)

# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser("ND2 flat-folder -> MERlin-style TIFFs using dataorganization + imageRegExp (no color_usage)")
    p.add_argument("--raw_data_path", required=True, help="Folder containing ND2 files (can be flat or nested)")
    p.add_argument("--dataorganization_csv", required=True, help="Path to dataorganization.csv")
    p.add_argument("--output_path", required=True, help="Where to write {imageType}_{fov}_{imagingRound}.tif")
    p.add_argument("--fov", type=int, default=None, help="Process only this fov (default: all)")

    # DeconWolf settings
    p.add_argument("--dw_path", required=True, help="Path to DeconWolf executable")
    p.add_argument("--ref_path", required=True, help="Folder with PSF tifs")
    p.add_argument("--decon_mode", choices=["none", "all", "readouts"], default="readouts",
                   help="none=skip decon, all=decon every frame, readouts=decon only frames referenced in dataorganization.frame")

    p.add_argument("--gpu", action="store_true")
    p.add_argument("--no_tile", action="store_true")
    p.add_argument("--tile_size", type=int, default=800)
    p.add_argument("--zstep", type=int, default=1200)
    p.add_argument("--n_iter", type=int, default=100)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true")

    args = p.parse_args()

    dataorg = load_dataorganization_csv(args.dataorganization_csv)
    filemap = build_filemap_from_dataorganization(args.raw_data_path, dataorg, allowed_exts=(".nd2",))

    export_merlin_tifs_from_nd2_flat(
        raw_data_path=args.raw_data_path,
        output_path=args.output_path,
        dataorg_df=dataorg,
        filemap_df=filemap,
        dw_path=args.dw_path,
        ref_path=args.ref_path,
        fov_filter=args.fov,
        decon_mode=args.decon_mode,
        gpu=args.gpu,
        tile=(not args.no_tile),
        tile_size=args.tile_size,
        zstep=args.zstep,
        n_iter=args.n_iter,
        scale=args.scale,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()
