import pandas as pd
import sweetviz as sv
import ast

df = pd.read_csv("data.csv")

report = sv.analyze(df)
report.show_html("report.html")