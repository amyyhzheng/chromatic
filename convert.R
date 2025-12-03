#!/usr/bin/env Rscript

# ---- User settings ----

# Path to your input RDS file
input_rds <- "/Volumes/weissman_imaging/amyzheng/datasets/fengoct25_merfish/GSE284005_merfish_all.rds"

# Directory where you want to save the h5Seurat and h5ad files
output_dir <- "/Volumes/weissman_imaging/amyzheng/datasets/merfish_h5"

# Base name (without extension) for output files
output_basename <- "GSE284005_merfish_all_dec2_nospat"


# ---- Setup ----

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratDisk)
})

cat("Reading RDS from:\n  ", input_rds, "\n")
obj <- readRDS(input_rds)

cat("Original images in object:\n")
print(names(obj@images))

# ---- Strip spatial images (FOV / Segmentation) to avoid SeuratDisk errors ----
# This keeps expression + metadata but drops spatial images.
cat("Removing all images from obj@images to avoid Segmentation/FOV issues...\n")
obj@images <- list()

cat("Images after removal:\n")
print(names(obj@images))

# ---- Save as .h5Seurat ----

h5seurat_path <- file.path(output_dir, paste0(output_basename, ".h5Seurat"))
cat("Saving h5Seurat to:\n  ", h5seurat_path, "\n")

SaveH5Seurat(
  obj,
  filename = h5seurat_path,
  overwrite = TRUE
)

cat("Finished SaveH5Seurat.\n")

# ---- Convert to .h5ad ----

cat("Converting h5Seurat to h5ad...\n")

Convert(
  h5seurat_path,
  dest = "h5ad",
  overwrite = TRUE
)

h5ad_path <- file.path(output_dir, paste0(output_basename, ".h5ad"))
cat("Done.\nOutput h5ad should be at:\n  ", h5ad_path, "\n")
