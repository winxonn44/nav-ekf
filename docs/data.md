# Datasets

## EuRoC MAV (machine hall)

The E4 results use the five machine-hall sequences of the EuRoC MAV dataset (DOI [10.3929/ethz-b-000690084](https://doi.org/10.3929/ethz-b-000690084)).
Its license is *In Copyright - Non-Commercial Use Permitted*, so **no dataset file or raw copy of one is in this repository**; only
code, configs and result summaries are.

To get the data, download `machine_hall.zip` through a browser from the ETH Research Collection page at that DOI. The site
rate-limits scripts, so a browser download is the expected route. Then extract only the files the code uses (IMU, ground truth and their
`sensor.yaml`, no images):

```bash
python -I scripts/extract_euroc.py /path/to/machine_hall.zip ~/Documents/Code/datasets/euroc --tmp /path/to/empty/dir
```

That writes `<data-root>/MH_0x_*/mav0/{imu0,state_groundtruth_estimate0}/{data.csv,sensor.yaml}` and `mav0/body.yaml`, about 40 MB in
all. It copies each nested zip to `--tmp` one at a time and deletes it, and it leaves the outer zip untouched. Treat downloaded data as untrusted: keep scripts outside the
data directory and run the tools with `python -I`.

Tests do not need this data: `python/navekf_py/euroc_fixture.py` writes a small synthetic sequence in the same layout.
