import numpy as np
import pandas as pd
import anndata as ad
import zarr
import numcodecs
from scipy.sparse import csr_matrix, csc_matrix

def generate_example(n_obs, n_var):
    # X and layers
    adata = ad.AnnData(csr_matrix(np.random.poisson(1, size=(n_obs, n_var)), dtype=np.float32))
    adata.layers["log_transformed"] = np.log1p(adata.X)
    adata.layers["other_data"] = np.random.poisson(1, size=(n_obs, n_var)) + 1.0

    # obs and var names
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]

    # annotations
    # the tutorial mentions that string arrays are automatically converted to categoricals if convenient
    adata.obs["cell_type"] = pd.Categorical(np.random.choice(["B", "T", "Monocyte"], size=(adata.n_obs,)))
    adata.obsm["X_umap"] = np.random.normal(0, 1, size=(adata.n_obs, 2))
    adata.varm["gene_stuff"] = np.random.normal(0, 1, size=(adata.n_vars, 5))
    adata.obsp["pairwise_data"] = csc_matrix(np.random.poisson(1, size=(n_obs, n_obs)), dtype=np.int32)

    # uns
    adata.uns["random"] = [1, 2, 3]

    return adata


def write_anndata(adata, filename, chunks):
    # root and tables
    root = zarr.open(filename, mode="w")
    root.array("some_image", np.array([0]), chunks=(1,))
    tables = root.create_group("tables")
    tables.attrs["tables"] = ["/anndata/obs", "/anndata/var", "/anndata/obsm", "/anndata/varm", "/anndata/obsp", "/anndata/varp"]
    adgroup = tables.create_group("anndata")
    adgroup.attrs["anndata"] = "0.9.1"
    adgroup.attrs["other-metadata"] = "metadata describing how the anndata annotates some_image"

    # X/layers
    X = adgroup.array('X', adata.X.todense(), chunks=chunks)
    layers = adgroup.create_group("layers")
    layers.array("log_transformed", adata.layers["log_transformed"].todense(), chunks=chunks)
    layers.array("other_data", adata.layers["other_data"], chunks=chunks)

    # obs
    localChunks = (chunks[0],)
    obs = adgroup.create_group("obs")
    obs.create_dataset("row_names", data=np.array(adata.obs_names), dtype=object, object_codec=numcodecs.VLenUTF8())
    obs.create_dataset("cell_type", data=np.array(adata.obs["cell_type"]), chunks=localChunks, object_codec=numcodecs.VLenUTF8())
    obs.attrs["annotated-data"] = get_annotated_data_map(dimension=0)
    obs.attrs["column-order"] = ["row_names", "cell_type"]

    # obsm
    obsm = adgroup.create_group("obsm")
    obsm.create_dataset("X_umap", data=adata.obsm["X_umap"], chunks=(chunks[0], 2))
    obsm.attrs["annotated-data"] = get_annotated_data_map(dimension=0)
    obsm.attrs["column-order"] = ["X_umap"]

    # obsp
    obsp = adgroup.create_group("obsp")
    obsp.create_dataset("pairwise_data", data=adata.obsp["pairwise_data"].todense(), chunks=(chunks[0], chunks[0]))
    obsp.attrs["annotated-data"] = get_annotated_data_map(dimension=0)
    obsp.attrs["column-order"] = ["pairwise_data"]

    # var
    var = adgroup.create_group("var")
    var.create_dataset("row_names", data=np.array(adata.var_names), chunks=(chunks[1],), object_codec=numcodecs.VLenUTF8())
    var.attrs["annotated-data"] = get_annotated_data_map(dimension=1)
    var.attrs["column-order"] = ["row_names"]

    # varm
    varm = adgroup.create_group("varm")
    varm.create_dataset("gene_stuff", data=adata.varm["gene_stuff"], chunks=(chunks[1], 5))
    varm.attrs["annotated-data"] = get_annotated_data_map(dimension=1)
    varm.attrs["column-order"] = ["gene_stuff"]

    # varm
    varp = adgroup.create_group("varp")
    varp.attrs["annotated-data"] = get_annotated_data_map(dimension=1)
    varp.attrs["column-order"] = []

    # uns
    uns = adgroup.create_group("uns")
    uns.create_dataset("random", data=adata.uns["random"], chunks=(3,))


def get_annotated_data_map(*, dimension):
    return [{"array": "/tables/anndata/X", "dimension": str(dimension)},
            {"array": "/tables/anndata/layers/log_transformed", "dimension": str(dimension)},
            {"array": "/tables/anndata/layers/other_data", "dimension": str(dimension)}]


def write_anndata_suggestion(adata, filename, chunks):
    # root and tables
    root = zarr.open(filename, mode="w")
    root.array("some_image", np.array([0]), chunks=(1,))
    tables = root.create_group("tables")
    tables.attrs["tables"] = ["/anndata/obs", "/anndata/var"]
    adgroup = tables.create_group("anndata")
    adgroup.attrs["anndata"] = "0.9.1"
    adgroup.attrs["other-metadata"] = "metadata describing how the anndata annotates some_image"

    # X and layers are combined into one array, a table is used to name the layers
    all_together = np.stack([np.array(adata.X.todense()), np.array(adata.layers["log_transformed"].todense()), adata.layers["other_data"]], axis=2)
    X = adgroup.array('X', all_together, chunks=(*chunks,1))
    layers = adgroup.create_group("layers")
    row_names = np.array(["X", "log_transformed", "other_data"])
    layers.create_dataset("row_names", data=row_names, dtype=object, object_codec=numcodecs.VLenUTF8())
    layers.attrs["annotated-data"] = [{"array": "/tables/anndata/X", "dimension": "2"}]

    # obs (combines obs, obsm, obsp)
    localChunks = (chunks[0],)
    obs = adgroup.create_group("obs")
    obs.create_dataset("row_names", data=np.array(adata.obs_names), dtype=object, object_codec=numcodecs.VLenUTF8())
    obs.create_dataset("cell_type", data=np.array(adata.obs["cell_type"]), chunks=localChunks, object_codec=numcodecs.VLenUTF8())
    obs.create_dataset("X_umap", data=adata.obsm["X_umap"], chunks=(chunks[0], 2))
    obs.create_dataset("pairwise_data", data=adata.obsp["pairwise_data"].todense(), chunks=(chunks[0], chunks[0]))
    obs.attrs["annotated-data"] = [{"array": "/tables/anndata/X", "dimension": "0"}]
    obs.attrs["column-order"] = ["row_names", "cell_type", "X_umap", "pairwise_data"]

    # var (combines var, varm, varp)
    var = adgroup.create_group("var")
    var.create_dataset("row_names", data=np.array(adata.var_names), chunks=(chunks[1],), object_codec=numcodecs.VLenUTF8())
    var.create_dataset("gene_stuff", data=adata.varm["gene_stuff"], chunks=(chunks[1], 5))
    var.attrs["annotated-data"] = [{"array": "/tables/anndata/X", "dimension": "1"}]
    var.attrs["column-order"] = ["row_names", "gene_stuff"]

    # uns
    uns = adgroup.create_group("uns")
    uns.create_dataset("random", data=adata.uns["random"], chunks=(3,))


# generate example and store as zarr using the minimal table spec proposal
n_obs = 10
n_var = 200
chunks = (10, 40)

adata = generate_example(n_obs, n_var)
write_anndata(adata, "example.zarr", chunks)

# store example in an alternative way, exploiting the properties of the suggested minimal table spec a bit more
write_anndata_suggestion(adata, "example_suggestion.zarr", chunks)
