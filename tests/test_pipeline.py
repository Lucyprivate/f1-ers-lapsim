import unittest
import numpy as np
from f1ers.cli import validate_data_gate,validate_telemetry
from f1ers.reporting import serializable,render_report
from pathlib import Path
import tempfile


class PipelineTests(unittest.TestCase):
    def test_lap_leakage_is_rejected(self):
        lap={"driver_number":1,"lap_number":2,"s_m":np.arange(200),"speed_mps":np.ones(200)}
        ds={"track":{"s_m":np.arange(200),"length_m":200},"laps":[{**lap,"role":r} for r in ["calibration","calibration","validation","validation"]]}
        with self.assertRaisesRegex(ValueError,"both calibration"):
            validate_data_gate(ds)

    def test_validation_never_reads_training_lap(self):
        n=200
        ds={"track":{"length_m":200},"laps":[{"driver_number":1,"lap_number":1,"role":"calibration","s_m":np.arange(n),"speed_mps":np.full(n,999),"lap_time_s":10},
              {"driver_number":1,"lap_number":2,"role":"validation","s_m":np.arange(n),"speed_mps":np.full(n,20),"lap_time_s":10}]}
        r={"s_m":np.arange(n),"speed_mps":np.full(n,20),"summary":{"lap_time_s":10}}
        v=validate_telemetry(ds,r)
        self.assertEqual(len(v["metrics"]),1)
        self.assertEqual(v["metrics"][0]["speed_rmse_kmh"],0)

    def test_json_embedding_escapes_script_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"report.html"
            render_report(path,{"limitations":["</script><script>alert(1)</script>"]})
            self.assertIn("\\u003c/script>",path.read_text(encoding="utf-8"))
        self.assertIsNone(serializable(np.nan))

    def test_sector_comparison_uses_solver_segment_times(self):
        s=np.arange(30)
        ds={"track":{"length_m":30},"laps":[{
            "driver_number":1,"lap_number":2,"role":"validation",
            "s_m":s,"speed_mps":np.full(30,10),"t_s":s*.2,
            "lap_time_s":6,"sector_times_s":[2,2,2]}]}
        r={"s_m":np.array([0,10,20]),"speed_mps":np.full(3,10),
           "dt_s":np.array([1,2,3]),"summary":{"lap_time_s":6}}
        metrics=validate_telemetry(ds,r)["metrics"][0]
        np.testing.assert_allclose(metrics["sim_sector_times_s"],[1,2,3])
        np.testing.assert_allclose(metrics["sector_error_s"],[-1,0,1])


if __name__=="__main__":unittest.main()
