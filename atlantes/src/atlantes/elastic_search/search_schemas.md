 Schema for the SUBPATH_INDEX in elastic_search_utils.py
{
  "activity_classification": [
    "unknown"
  ],
  "activity_details.confidence": [
    0.79895675
  ],
  "activity_details.model": [
    "ATLAS-Activity-Real-Time_no_git_hash_2024-09-06-19-56-12_epoch2.pt"
  ],
  "activity_details.model_version": [
    "b64c5a8_2024-10-08_10-05-08"
  ],
  "activity_details.model_version.keyword": [
    "b64c5a8_2024-10-08_10-05-08"
  ],
  "activity_details.model.keyword": [
    "ATLAS-Activity-Real-Time_no_git_hash_2024-09-06-19-56-12_epoch2.pt"
  ],
  "activity_details.original_classification": [
    "transiting"
  ],
  "activity_details.original_classification.keyword": [
    "transiting"
  ],
  "activity_details.outputs": [
    0.19790538,
    0.002513076,
    0.00062475016,
    0.79895675
  ],
  "activity_details.params.end_time": [
    "2024-10-08T16:59:25.000Z"
  ],
  "activity_details.params.num_positions": [
    2341
  ],
  "activity_details.params.position_lower_limit": [
    1000
  ],
  "activity_details.params.position_upper_limit": [
    3000
  ],
  "activity_details.params.start_time": [
    "2024-07-10T16:59:25.000Z"
  ],
  "activity_details.params.vessel.category": [
    79
  ],
  "activity_details.params.vessel.flag": [
    "ARE"
  ],
  "activity_details.params.vessel.flag.keyword": [
    "ARE"
  ],
  "activity_details.params.vessel.name": [
    "TARFFAH 1"
  ],
  "activity_details.params.vessel.name.keyword": [
    "TARFFAH 1"
  ],
  "activity_details.postprocessed_classification": [
    "unknown"
  ],
  "activity_details.postprocessed_classification.keyword": [
    "unknown"
  ],
  "cog": [
    274.9
  ],
  "created": [
    "2024-10-08T17:16:32.197Z"
  ],
  "end_location": [
    {
      "coordinates": [
        53.02945,
        25.10592
      ],
      "type": "Point"
    }
  ],
  "end_time": [
    "2024-10-08T16:59:26.000Z"
  ],
  "id": [
    "2c4c5d07-3bdf-4f6e-ac5a-bb4a503032e4"
  ],
  "mean_sog": [
    6.18
  ],
  "midpoint_cog": [
    176.08072
  ],
  "midpoint_geometry": [
    {
      "coordinates": [
        53.033244999999994,
        25.10566
      ],
      "type": "Point"
    }
  ],
  "mmsi": [
    "470159000"
  ],
  "num_positions": [
    6
  ],
  "path_geometry": [
    {
      "type": "GeometryCollection",
      "geometries": [
        {
          "coordinates": [
            53.03704,
            25.1054
          ],
          "type": "Point"
        },
        {
          "coordinates": [
            [
              53.03704,
              25.1054
            ],
            [
              53.02945,
              25.10592
            ]
          ],
          "type": "LineString"
        }
      ]
    }
  ],
  "start_location": [
    {
      "coordinates": [
        53.03704,
        25.1054
      ],
      "type": "Point"
    }
  ],
  "start_time": [
    "2024-10-08T16:55:32.000Z"
  ],
  "track_id": [
    "B:470159000:1633351597:2344594:1143540"
  ],
  "updated": [
    "2024-10-08T17:26:34.769Z"
  ],
  "_id": "2c4c5d07-3bdf-4f6e-ac5a-bb4a503032e4",
  "_index": "subpath-000010",
  "_score": null
}


Example of th SEARCH_HISTORY_INDEX schema

{
  "created": [
    "2024-10-14T20:58:28.894Z"
  ],
  "end.point": [
    {
      "coordinates": [
        -93.80048,
        29.55621
      ],
      "type": "Point"
    }
  ],
  "end.time": [
    "2024-10-14T20:37:41.000Z"
  ],
  "event_details.fishing_score": [
    0.822409
  ],
  "event_details.model_name": [
    "ATLAS-Activity-Real-Time_no_git_hash_2024-09-06-19-56-12_epoch2.pt"
  ],
  "event_details.model_name.keyword": [
    "ATLAS-Activity-Real-Time_no_git_hash_2024-09-06-19-56-12_epoch2.pt"
  ],
  "event_details.model_version": [
    "35b89d4_2024-10-14_18-00-57"
  ],
  "event_details.model_version.keyword": [
    "35b89d4_2024-10-14_18-00-57"
  ],
  "event_details.params.end_time": [
    "2024-10-14T20:37:40.000Z"
  ],
  "event_details.params.num_positions": [
    2188
  ],
  "event_details.params.position_lower_limit": [
    100
  ],
  "event_details.params.position_upper_limit": [
    2048
  ],
  "event_details.params.start_time": [
    "2024-09-14T20:37:40.000Z"
  ],
  "event_details.params.vessel.category": [
    9999
  ],
  "event_details.params.vessel.flag": [
    "USA"
  ],
  "event_details.params.vessel.flag.keyword": [
    "USA"
  ],
  "event_details.params.vessel.name": [
    "GALVESTON ISLAND"
  ],
  "event_details.params.vessel.name.keyword": [
    "GALVESTON ISLAND"
  ],
  "event_details.subpath_ids": [
    "a85096f9-1dee-4993-96fc-65803a4c25e4"
  ],
  "event_details.subpath_ids.keyword": [
    "a85096f9-1dee-4993-96fc-65803a4c25e4"
  ],
  "event_id": [
    "de7428f1-4be1-4cee-8d98-659ee853e79d"
  ],
  "event_type": [
    "fishing_activity_history"
  ],
  "latest.point": [
    {
      "coordinates": [
        -93.80048,
        29.55621
      ],
      "type": "Point"
    }
  ],
  "latest.time": [
    "2024-10-14T20:37:41.000Z"
  ],
  "start.point": [
    {
      "coordinates": [
        -93.80134,
        29.55598
      ],
      "type": "Point"
    }
  ],
  "start.time": [
    "2024-10-14T20:30:10.000Z"
  ],
  "updated": [
    "2024-10-14T20:58:28.894Z"
  ],
  "vessel_count": [
    1
  ],
  "vessels.vessel_0.attribution.display_country": [
    "AIS"
  ],
  "vessels.vessel_0.attribution.display_country.keyword": [
    "AIS"
  ],
  "vessels.vessel_0.attribution.display_name": [
    "AIS"
  ],
  "vessels.vessel_0.attribution.display_name.keyword": [
    "AIS"
  ],
  "vessels.vessel_0.attribution.mmsi": [
    "AIS"
  ],
  "vessels.vessel_0.attribution.mmsi.keyword": [
    "AIS"
  ],
  "vessels.vessel_0.category": [
    "unknown"
  ],
  "vessels.vessel_0.class": [
    "vessel"
  ],
  "vessels.vessel_0.country_filter": [
    "USA"
  ],
  "vessels.vessel_0.display_country": [
    "USA"
  ],
  "vessels.vessel_0.display_name": [
    "GALVESTON ISLAND"
  ],
  "vessels.vessel_0.mmsi": [
    366304000
  ],
  "vessels.vessel_0.name": [
    "GALVESTON ISLAND"
  ],
  "vessels.vessel_0.track_id": [
    "B:366304000:1682703053:888771:1196538"
  ],
  "vessels.vessel_0.type": [
    "unknown"
  ],
  "vessels.vessel_0.vessel_id": [
    "B:366304000:1682703053:888771:1196538"
  ],
  "_id": "de7428f1-4be1-4cee-8d98-659ee853e79d",
  "_index": "event-history",
  "_score": null
}
