# Dataset Creation

The scripts are used to create the raw training data for ais based modeling and creates an end product of monthly tracks for a single trackid. The initial data source is either gcs bucket `skylight-data-sky-prod-a-dayc/track/track-incremental` or `skylight-data-sky-int-a-wxbc/track/track-incremental` which stores Dynamic AIS data in avro files. Additionaly, static AIS data which includes vessel metadata such as vessel name, ais type etc.


Note: TrackIds in production and integration may differ so only use either prod or int data for all the steps.




## Steps to Build Training dataset of TrackId Month Files

### 1. Pull the raw Dynamic and Static AIS data from GCS
1. Download daily track incremental to disk
    $ gcloud alpha storage cp -r gs://skylight-data-sky-int-a-wxbc/track/track-incremental/\
track-incremental.2022-*.avro ./track-incremental
2. Transform Daily track incremental to csv
    $ track_avro_to_csv.py
3. gsutil cp $(gsutil ls -l "$(gsutil ls gs://skylight-data-sky-int-a-wxbc/vessel/vessel-snapshot/ | sort -r | head -1)*.avro" | sort -k2n | tail -1 | awk '{print $NF}') . (Pull latest vessel snapshot, post 03/2024 there will be less duplicates)
4. Generate static vessel metadata_csv
    $ gen_vessel_metadata_csv.py

Outputs: Daily track csvs with 1 record per row, csv with metadata associated to `trackId`

### 2. Build Training Dataset for AIS Modeling for a given years


Assumes you have a vessel_metadata.csv and a directory of track incremental csvs, want to use at least a 96 CPU machine for this.

1. Create a parquet file with all the unique trackIds in each month to aid with future processing
`python3 gen_monthly_unique_trackid_parquet.py`
2. Create a directory containing parquet files for each day for each unique trackid in a directory structure of {ais_type}/{flag}/{int}/{trackid}/{trackid_month}/trackid_month_day.parquet
   ` python3 track_incremental_to_daily_trackid_parquet.py --path-to-input-csvs /path/to/track-incremental-csvs --path-to-vessel-metadata /path/to/vessel-metadata.csv --output-training-dir-root /path/to/output-dir`
3. Generate local gcloud auth credentials and update the path in the gen_subpaths_training.py (Will make this more flexible in the future)
    `gcloud auth application-default login`
4. Copy from local disk to gcloud bucket
    `gcloud alpha storage cp -r path/to/data gs://ais-track-data/{year}/{directory}`
5. Create a version of the dataset with precomputed subpath_num columns which indicate activity change and our used as the unit on which we classify (also combines daily parquets into monthly)
` python3 gen_cpd_subpaths.py --root-dir /data-mount/ais_data_{year}_meta/ --output-dir /data-mount/all_{year}_tracks_w_subpaths`
6. Copy from local disk to gcloud bucket (ENSURE DIFFERENT name then non subpath data)
    `gcloud alpha storage cp -r path/to/data gs://ais-track-data`
7. (optional) Create metadata parquet for the year: It may be useful for other data procesing task to have associations to differen pieces of metadata via a parquet loadable as a dataframe so we can eaisly query by vessel type month etc. (this code assumes that metadata was written into the path)
 `create_metadata_index_parquet`


Outputs: We should have a directory in a gcs bucket with trackID month parquets

## Generating Labeled Datasets

### AIS Behvaior Classification
1.  From Human Annotation
    - see the docs in `atlantes/human annotation` for the workflow for sampling, annotating and exporting the data
2.  From Machine Annotation
    - see the docs in `atlantes/machine_annotation` for the workflow for generating the data


For End of Sequence Prediction:

`create_end_of_seq_activity_dataset.py`

For dense prediction:

`create_ais_activity_dataset.py`


### Buoy Vessel Classification

We use metadata and SME heuristics to generate labels for the dataset.

`gen_buoy_vessel_annotations.py`

### Vessel Type Classification

We use reported static AIS data to generate labels for the dataset. About 40% of vessels do not report a vessel type.

`create_entity_vessel_type_labels.py`

### One Sided Rendezvous Classification

TBD currently we have an OSR model branch but it has not been merged or updated to use the end of sequence task set up

### Buoy Machine Annotation Assumptions

1. Phrases like Buoy, Net, Fishing Gear followed by numbers and/or percents is likely to give buoys
2. Only English Speaking Countries would have Buoy Pun Vessel names
3. MMSI starting with 0,8,9 are more likely to be ambigously identified
4. Everything with int percent is a buoy
