import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

source=Path(__file__).resolve().parents[1]/"experiments/04_event_phase_analysis.py"
spec=importlib.util.spec_from_file_location("event_analysis",source)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class PhaseRules(unittest.TestCase):
    def test_six_observed_dry_hours_split_events(self):
        rain=pd.Series(0.,index=pd.date_range("2022-01-01",periods=160,freq="h"))
        rain.iloc[[10,16,23]]=[1.,1.,1.]
        events,ids=mod.catalog_events(rain)
        self.assertEqual(ids.dropna().tolist(),[1,1,2])
        self.assertEqual(events.n_wet_hours.tolist(),[2,1])
        self.assertEqual(events.previous_dry_hours.iloc[1],6)

    def test_phase_endpoints_and_current_four_hour_window(self):
        rain=pd.Series(0.,index=pd.date_range("2022-01-01",periods=160,freq="h"))
        rain.iloc[63]=1.
        _,ids=mod.catalog_events(rain)
        labels=mod.phase_labels(rain,ids)
        for i in [63,64,65,66]:self.assertEqual(labels.phase.iloc[i],"raining")
        for i,expected in [(67,"post_0_12h"),(75,"post_0_12h"),(76,"post_12_24h"),
                           (87,"post_12_24h"),(88,"post_24_48h"),(111,"post_24_48h"),(112,"dry_over48h")]:
            self.assertEqual(labels.phase.iloc[i],expected)

    def test_missing_weather_is_not_dry_and_new_rain_resets_clock(self):
        rain=pd.Series(0.,index=pd.date_range("2022-01-01",periods=160,freq="h"))
        rain.iloc[63]=1.;rain.iloc[80]=np.nan
        _,ids=mod.catalog_events(rain)
        labels=mod.phase_labels(rain,ids)
        self.assertEqual(labels.phase.iloc[88],"unknown")
        self.assertEqual(labels.phase.iloc[112],"unknown")
        self.assertEqual(labels.phase.iloc[129],"dry_over48h")
        rain.iloc[90]=.2
        _,ids=mod.catalog_events(rain)
        labels=mod.phase_labels(rain,ids)
        self.assertEqual(labels.phase.iloc[94],"post_0_12h")
        self.assertEqual(labels.hours_since_last_wet_hour.iloc[94],4.)


if __name__=="__main__":unittest.main()
