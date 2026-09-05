"""theta_zoom: alias of `rung` (the tool's name through release 1.1.0).

Kept importable for one release so that `from theta_zoom import zoom` and the
`theta-zoom` command keep working. New code should import `rung`.
"""
from rung import *  # noqa: F401,F403
from rung import main, zoom, llm_battery, build_axis, axis_from_dataset, summarize, summarize_data  # noqa: F401

if __name__ == "__main__":
    main()
