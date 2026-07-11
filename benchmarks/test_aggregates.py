# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import json
import pytest
from test_bench_base import TestBenchBase
from sedonadb.testing import DuckDB, PostGIS, SedonaDB

POINTS_PER_GROUP = [8, 100, 1000]


# Each stage adds one operation to the stage above it, so the difference between two
# consecutive stages is the cost of the operation that was added. The last stage is the
# equivalent aggregate, measured against collect_hull_area. ST_ConvexHull_Agg has no
# PostGIS/DuckDB equivalent, so it only runs on SedonaDB.
def stage_projection(eng, stage):
    if isinstance(eng, PostGIS):
        # PostGIS aggregate functions don't have a _Agg suffix
        collect = "ST_Collect(geometry)"
        envelope_agg = "ST_Envelope(ST_Collect(geometry))"
    elif isinstance(eng, DuckDB):
        # ST_Collect is a scalar over a list(geometry) aggregate
        collect = "ST_Collect(list(geometry))"
        envelope_agg = "ST_Envelope_Agg(geometry)"
    else:
        collect = "ST_Collect_Agg(geometry)"
        envelope_agg = "ST_Envelope_Agg(geometry)"

    return {
        "count": "COUNT(*)",
        "envelope_agg": envelope_agg,
        "collect_agg": collect,
        "collect_hull": f"ST_ConvexHull({collect})",
        "collect_hull_area": f"ST_Area(ST_ConvexHull({collect}))",
        "convexhull_agg_area": "ST_Area(ST_ConvexHull_Agg(geometry))",
    }[stage]


STAGE_ENGINES = {
    "count": [SedonaDB, PostGIS, DuckDB],
    "envelope_agg": [SedonaDB, PostGIS, DuckDB],
    "collect_agg": [SedonaDB, PostGIS, DuckDB],
    "collect_hull": [SedonaDB, PostGIS, DuckDB],
    "collect_hull_area": [SedonaDB, PostGIS, DuckDB],
    "convexhull_agg_area": [SedonaDB],
}
STAGE_ENGINE_PAIRS = [
    (stage, eng) for stage, engines in STAGE_ENGINES.items() for eng in engines
]


class TestBenchConvexHullAggStages(TestBenchBase):
    def setup_class(self):
        """Setup test data for grouped convex hull benchmarks"""
        self.sedonadb = SedonaDB.create_or_skip()
        self.postgis = PostGIS.create_or_skip()
        self.duckdb = DuckDB.create_or_skip()

        self.num_rows = 100_000

        point_options = {
            "geom_type": "Point",
            "num_rows": self.num_rows,
            "seed": 42,
        }

        point_query = f"""
            SELECT id, geometry
            FROM sd_random_geometry('{json.dumps(point_options)}')
        """
        point_tab = self.sedonadb.execute_and_collect(point_query)
        self.sedonadb.create_table_arrow("hull_points", point_tab)
        self.postgis.create_table_arrow("hull_points", point_tab)
        self.duckdb.create_table_arrow("hull_points", point_tab)

    @pytest.mark.parametrize("stage,eng", STAGE_ENGINE_PAIRS)
    @pytest.mark.parametrize("points_per_group", POINTS_PER_GROUP)
    def test_convex_hull_agg_stages(self, benchmark, points_per_group, stage, eng):
        """Benchmark each stage of the grouped convex hull chain"""
        eng = self._get_eng(eng)
        num_groups = self.num_rows // points_per_group

        def queries():
            eng.execute_and_collect(f"""
                SELECT {stage_projection(eng, stage)}
                FROM hull_points
                GROUP BY id % {num_groups}
            """)

        benchmark(queries)
