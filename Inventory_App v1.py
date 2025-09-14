import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import streamlit as st
import json

# Load credentials from Streamlit secrets
creds_dict = st.secrets["gcp_service_account"]
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)

def load_sheet(tab_name):
    sheet = client.open("BusinessData").worksheet(tab_name)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def save_sheet(tab_name, df):
    sheet = client.open("BusinessData").worksheet(tab_name)
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())
# ----------------------------
# Load all tabs
# ----------------------------
purchases_df = load_sheet("Purchases")
sales_df = load_sheet("Sales")
kits_df = load_sheet("Kits")
defects_df = load_sheet("Defects")
inventory_df = load_sheet("Inventory")

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Inventory Management App")

# ----------------------------
# Add Purchase
# ----------------------------
st.subheader("Add New Purchase")
col1, col2, col3 = st.columns(3)
item = col1.text_input("Item Name")
qty = col2.number_input("Quantity", min_value=1)
price = col3.number_input("Price per Unit", min_value=0.0)

if st.button("Add Purchase"):
    if item:
        new_row = {"Item": item, "Quantity": qty, "Price": price}
        purchases_df = pd.concat([purchases_df, pd.DataFrame([new_row])], ignore_index=True)
        save_sheet("Purchases", purchases_df)
        st.success(f"Added {qty} x {item} to Purchases")
    else:
        st.error("Please enter an Item Name")

# ----------------------------
# Add Sale
# ----------------------------
st.subheader("Add New Sale")
col1, col2 = st.columns(2)
sold_item = col1.selectbox("Select Item", inventory_df["Item"].tolist())
sold_qty = col2.number_input("Quantity Sold", min_value=1)

if st.button("Add Sale"):
    stock_row = inventory_df[inventory_df["Item"] == sold_item]
    if stock_row.empty:
        st.error("Item not in inventory")
    elif sold_qty > int(stock_row["Quantity"].values[0]):
        st.error("Not enough stock!")
    else:
        new_sale = {"Item": sold_item, "Quantity": sold_qty}
        sales_df = pd.concat([sales_df, pd.DataFrame([new_sale])], ignore_index=True)
        save_sheet("Sales", sales_df)
        
        # Update inventory
        inventory_df.loc[inventory_df["Item"] == sold_item, "Quantity"] -= sold_qty
        save_sheet("Inventory", inventory_df)
        st.success(f"Recorded sale of {sold_qty} x {sold_item}")

# ----------------------------
# View Inventory
# ----------------------------
st.subheader("Current Inventory")
st.dataframe(inventory_df)

# ----------------------------
# Add Defect
# ----------------------------
st.subheader("Add Defective Item")
def_item = st.selectbox("Select Item (Defective)", inventory_df["Item"].tolist())
def_qty = st.number_input("Quantity Defective", min_value=1)

if st.button("Mark as Defective"):
    stock_row = inventory_df[inventory_df["Item"] == def_item]
    if def_qty > int(stock_row["Quantity"].values[0]):
        st.error("Not enough stock to mark defective")
    else:
        new_defect = {"Item": def_item, "Quantity": def_qty}
        defects_df = pd.concat([defects_df, pd.DataFrame([new_defect])], ignore_index=True)
        save_sheet("Defects", defects_df)
        
        # Update inventory
        inventory_df.loc[inventory_df["Item"] == def_item, "Quantity"] -= def_qty
        save_sheet("Inventory", inventory_df)
        st.success(f"Marked {def_qty} x {def_item} as defective")

# ----------------------------
# Kits Management
# ----------------------------
st.subheader("Manage Kits")
kit_name = st.text_input("Kit Name")
kit_items = st.text_area("Items (comma separated, e.g., Item1:2,Item2:1)")

if st.button("Add Kit"):
    if kit_name and kit_items:
        new_kit = {"Kit Name": kit_name, "Items": kit_items}
        kits_df = pd.concat([kits_df, pd.DataFrame([new_kit])], ignore_index=True)
        save_sheet("Kits", kits_df)
        st.success(f"Added Kit: {kit_name}")
    else:
        st.error("Enter kit name and items")

# ----------------------------
# View Purchases, Sales, Defects, Kits
# ----------------------------
st.subheader("Purchase History")
st.dataframe(purchases_df)

st.subheader("Sales History")
st.dataframe(sales_df)

st.subheader("Defective Items")
st.dataframe(defects_df)

st.subheader("Kits")
st.dataframe(kits_df)
