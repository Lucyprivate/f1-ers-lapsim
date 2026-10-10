import unittest
from pathlib import Path
import numpy as np

from f1ers_v3.geometry import ClosedRoute,build_track,remap_rules
from f1ers_v3.protocol import read_json

ROOT=Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
    def circle(self):
        angle=np.linspace(0,2*np.pi,96,endpoint=False)
        return ClosedRoute(np.column_stack([50*np.cos(angle),50*np.sin(angle)]),kind="spline",phase=3.7)

    def test_periodic_seam_position_tangent_curvature(self):
        route=self.circle()
        sample=route.evaluate(np.array([0.,route.length_m]))
        for key in ("x_m","y_m","curvature_1pm","tangent"):
            np.testing.assert_allclose(sample[key][0],sample[key][1],atol=1e-11)
        self.assertLess(abs(route.length_m-100*np.pi),.001)
        self.assertLess(abs(float(sample["curvature_1pm"][0])-.02),2e-5)

    def test_mesh_resampling_never_refits_curve(self):
        route=self.circle()
        sample=[route.sample(d) for d in (10,5,2.5)]
        self.assertEqual(len({x["curve_id"] for x in sample}),1)
        self.assertEqual(len({x["length_m"] for x in sample}),1)
        for mesh in sample:
            np.testing.assert_allclose(mesh["x_m"],route.evaluate(mesh["s_m"])["x_m"],atol=0)
            self.assertLessEqual(mesh["ds_m"],mesh["requested_ds_m"])
            self.assertAlmostEqual(float(mesh["segment_ds_m"].sum()),route.length_m)

    def test_source_parameter_roundtrip_and_direction(self):
        route=self.circle()
        u=np.linspace(.1,route.source_length-.1,101)
        np.testing.assert_allclose(route.s_to_source(route.source_to_s(u)),u,atol=2e-7)
        backwards=ClosedRoute(route.points,kind="spline",phase=route.phase,direction=-1)
        np.testing.assert_allclose(backwards.evaluate(0)["tangent"],-route.evaluate(0)["tangent"],atol=1e-12)
        self.assertLess(float(backwards.evaluate(0)["curvature_1pm"]),0)

    def test_real_route_boundaries_and_same_rule_mapping(self):
        for candidate in ("alignment","spline","anisotropic"):
            track=build_track(ROOT,candidate,5.)
            self.assertGreater(track["geometry_audit"]["minimum_boundary_clearance_m"],0)
            rules=read_json(ROOT/"config/rules_austria_2026.json")
            mapped=remap_rules(rules,track)
            route=track["continuous_route"]
            for (a,b),(c,d) in zip(rules["alternative_power_zones_m"],mapped["alternative_power_zones_route_m"]):
                self.assertAlmostEqual(c,float(route.legacy_to_s(a*route.source_length/4326)))
                self.assertAlmostEqual(d,float(route.legacy_to_s(b*route.source_length/4326)))
            self.assertFalse(mapped["coordinate_transform"]["surveyed"])


if __name__=="__main__":unittest.main()
