{
  "alert": "CRITICAL ANOMALY DETECTED",
  "timestamp": 1728253317000,
  "asset": {
    "node_id": "gw_extraccion_112",
    "device_id": "plc_siemens_lab"
  },
  "health_assessment": {
    "health_score_percent": 0.0,
    "namur_status": "CRITICAL",
    "deviation_pct": 744.71
  },
  "root_cause_analysis": [
    {
      "sensor": "FIT_001_VOL",
      "error_raw": 575.99,
      "error_rel_pct": 45.2
    },
    {
      "sensor": "FIT_001_MAS",
      "error_raw": 532.16,
      "error_rel_pct": 41.8
    }
  ]
}


{
  "timestamp": 1728253317000,
  "asset": {
    "node_id": "gw_extraccion_112",
    "device_id": "plc_siemens_lab"
  },
  "metrics": {
    "health_score_percent": 98.5,
    "namur_status": "OPTIMAL",
    "mse_raw": 1.3757,
    "threshold_p95": 9.3637
  }
}
