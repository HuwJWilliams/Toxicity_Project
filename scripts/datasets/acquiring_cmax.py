# %%
import xml.etree.ElementTree as ET
import pandas as pd

tree = ET.parse("drugbank_full_data.xml")
root = tree.getroot()


# %%
