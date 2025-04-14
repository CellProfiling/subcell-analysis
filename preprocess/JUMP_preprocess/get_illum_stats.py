import click
import pandas as pd
import os
from preprocess.illumination_statistics import calculate_statistics
import multiprocessing



@click.command()
@click.option('--metadata_csv', required=True, type=click.Path(exists=True))
@click.option('--outdir', required=True)
def get_illum_stats(metadata_csv, outdir):
    
    #Make separate DataFrames for plates
    metadata = pd.read_csv(metadata_csv)
    plates = metadata["Plate"].unique()
    plate_dfs = [metadata[metadata["Plate"] == plate].reset_index(drop=True) for plate in plates]
    outfiles = [f"{outdir}/{plate}_illumstats.pkl" for plate in plates]
    params = zip(plate_dfs, plates, outfiles)


    print(f"Num of CPUS: {multiprocessing.cpu_count()}")
    os.makedirs(outdir, exist_ok=True)
    with multiprocessing.Pool() as p:
        p.starmap(calculate_statistics, params)
    
if __name__ == '__main__':
    get_illum_stats()
