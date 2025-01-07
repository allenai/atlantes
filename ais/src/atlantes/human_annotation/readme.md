# Steps to generate a dataset for annotating (for ATLAS)
Make sure you have an Elastic Search account and set the environment variables for SEARCH_USERNAME and SEARCH_PASSWORD

1. Generate Monthly Fishing Traffic geotiffs using 'visualize_monthly_fishing_traffic.py'
2. Upload the geotiffs to google drive in 'Marine Maps folder'
3. Label AOIs via SMEs in qgis (manual)
   a. This involves looking at the geotiffs and making boxes of areas that seem to have a lot of fishing activity throughout the year (labeling boxes per month to get seasonal distirbution) these geojsons should be in '{year}_{month}_{ais_category}_areas.geojson
4. Download geojsons from google drive and add to 'fishing_areas' folder
5. Stratified sample across month and ai with 'sample_from_fishing_regions.py'
   a. Or sample using sample_hard_negatives.py
6. Generate data from a .txt of sampled tracks using  `create_downsampled_training_data.py``
7.
   a. Delete the old parquets from within the output_dir 'find . -type f -name "*.parquet" -exec rm -f {} \;'
   b. Copy that data: https://console.cloud.google.com/storage/browser/skylight-data-sky-int-a-wxbc/annotations/new?project=skylight-int-a-r2d2 The data must be in the structure of `annotations/new/{project_name}/{trajectory_name}.csv`
   Use command tool `gsutil -m cp -r {local_path} gs://skylight-data-sky-int-a-wxbc/annotations/new` for integration
   Use command tool `gsutil -m cp -r {local_path} gs://skylight-data-sky-prod-a-dayc/annotations/new` for prod-a
8. After the data is uploaded, run the `ap_annotation_builder` airflow job.:
   - **Integration**: https://t8706d9aa464274cfp-tp.appspot.com/admin/airflow/tree?dag_id=ap_annotation_builder
   - **Prod-A**: https://f5581c439208469c8a1bbb319e31df0b-dot-us-west1.composer.googleusercontent.com/dags/ap_annotation_builder/grid?search=ap_annotation_builder
9. (Annotate)
10. Using the [`mda_annotation_exporter`](https://f5581c439208469c8a1bbb319e31df0b-dot-us-west1.composer.googleusercontent.com/dags/mda_annotation_exporter/grid?search=mda_annotation_exporter ) Airflow job. When you trigger the DAG you'll need to provide a JSON configuration of the projects you want to export `{"projects": "projectA,projectB,projectC"}`.
   - The exported annotations will show up in GCS in `annotations/output/{project_name}/{trajectory_name}.csv`
11. Create the final data product for atlas activity detection. This is done by using the create_ais_activity_dataset.py script or the equivalent script for the end of sequence activity prediction. This script will:
   - Join extracted data (from 10) to original data (not the downsampled trajectories)
      -note: not all subpaths have activity labels after downsampling so see the upsample_human_annotations.py script for details
   - Optionally add machine annotate non fishing vessel tracks of known type
   - Exclude tracks in projects we have flagged in HA as needing to be excluded
   - Advisised to run on a bigger machine
   - The ouput will be placed on gcp in the 'ais-track-data' bucket in the {year}/labels:
      - a csv to load an End of Sequence Activity Dataset


_Note_: If prod-b is active, then you'll need to ask Hunter or Josh to copy the annotation projects from prod-a over to prod-b. Here's the command they need to run:

```
# Note: you must add your username:password to these urls.
elasticdump --input=https://skylight-prod-a.es.us-west1.gcp.cloud.es.io:9243/track_annotation_summaries --output=https://skylight-prod-b.es.us-west1.gcp.cloud.es.io:9243/track_annotation_summaries --type=data
```
