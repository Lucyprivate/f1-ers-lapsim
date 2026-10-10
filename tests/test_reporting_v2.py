import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from f1ers.reporting_v2 import export_trace_v2

TMP=Path(__file__).resolve().parents[3]/"work"/"test_tmp"
TMP.mkdir(parents=True,exist_ok=True)


class TraceV2Tests(unittest.TestCase):
    def test_export_contains_terminal_state_and_no_invented_terminal_power(self):
        result={"s_m":np.array([0,10]),"x_m":np.array([1,2]),"y_m":np.array([3,4]),
            "speed_mps":np.array([20,25]),"dt_s":np.array([.4,.5]),"time_s":np.array([0,.4,.9]),
            "mguk_dc_kw":np.array([-100,50]),"energy_mj":np.array([2,2.04,2.015]),
            "guard_reasons":[[],["reserve","cap"]],"summary":{"completed_lap":True,"speed_end_mps":20}}
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            path=Path(folder)/"trace.csv";export_trace_v2(path,result)
            with path.open(encoding="utf-8-sig",newline="") as handle:rows=list(csv.DictReader(handle))
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[-1]["row_type"],"terminal")
        self.assertEqual(float(rows[-1]["energy_mj"]),2.015)
        self.assertEqual(float(rows[-1]["time_s"]),.9)
        self.assertEqual(rows[-1]["mguk_dc_kw"],"")
        delta=sum(float(r["mguk_dc_kw"])*float(r["dt_s"])/1000 for r in rows[:-1])
        self.assertAlmostEqual(float(rows[-1]["energy_mj"])-float(rows[0]["energy_mj"])+delta,0)


if __name__=="__main__":unittest.main()
