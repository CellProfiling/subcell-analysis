import click
from preprocess.preprocess import preprocess
from functools import partial
import pandas as pd
import multiprocessing
import os


def preprocess_plate(metadata, csv_name, crop_size, save_dir, save_crops=True, center_type="cell", reduce=True):
    df = preprocess(metadata, crop_size, save_dir, save_crops=True, center_type="cell", reduce=reduce)
    df.to_csv(csv_name)
    return df

@click.command()
@click.option('--metadata_csv', required=True, type=click.Path(exists=True))
@click.option('--csv_dir', required=True, type=str)
@click.option('--crop_size', required=True, type=int)
@click.option('--save_dir', required=True)
@click.option('--save_crops', default=False, is_flag=True)
@click.option('--reduce', default=False, is_flag=True)
def make_crops(metadata_csv, csv_dir, crop_size, save_dir, save_crops, reduce):
    
    assert crop_size %2 == 0
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    #preprocess(metadata_csv, crop_size, save_dir, save_crops=save_crops, reduce=reduce)

    #Make separate DataFrames for plates
    metadata = pd.read_csv(metadata_csv)
    #metadata = pd.read_csv(metadata_csv).sample(10).reset_index()
    plates = metadata["Plate"].unique()
    plate_dfs = [metadata[metadata["Plate"] == plate].reset_index(drop=True) for plate in plates]
    csv_names = [f"{csv_dir}/{plate}_centers_test.csv" for plate in plates]
    params = zip(plate_dfs, csv_names)

    #Partially evaluate preprocess function with fixed arguments
    f = partial(preprocess_plate, crop_size=crop_size, save_dir=save_dir, save_crops=save_crops, reduce=reduce)
    #plate_dfs = [(df,) for df in plate_dfs] #iterable format for starmap

    print(f"Num of CPUS: {multiprocessing.cpu_count()}")
    with multiprocessing.Pool() as p:
        #dfs = p.map(f, plate_dfs)
        p.starmap(f, params)

    #pd.concat(dfs).to_csv(csv_name, index=False)

if __name__ == '__main__':
    make_crops()
