# inventory_app.py
import os
from datetime import datetime, date
from typing import List, Tuple

import pandas as pd
import streamlit as st

# ----------------------------
# Config & files
# ----------------------------
st.set_page_config(page_title="Inventory & Kits Manager", page_icon="📦", layout="wide")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    "purchases": os.path.join(DATA_DIR, "purchases.csv"),
    "master": os.path.join(DATA_DIR, "master_inventory.csv"),   # derived snapshot (we still save snapshot for easy export)
    "kits_bom": os.path.join(DATA_DIR, "kits_bom.csv"),
    "created_kits": os.path.join(DATA_DIR, "created_kits.csv"),
    "sold_kits": os.path.join(DATA_DIR, "sold_kits.csv"),
    "defective": os.path.join(DATA_DIR, "defective_items.csv"),
    "restock": os.path.join(DATA_DIR, "restock_items.csv"),
}

# ----------------------------
# DEFAULT (templates)
# ----------------------------
DEFAULTS = {
    "purchases": pd.DataFrame(columns=[
        "Date", "Material ID", "Material Name", "Vendor", "Packs", "Qty Per Pack", "Cost Per Pack", "Pieces", "Price Per Piece"
    ]),
    "master": pd.DataFrame(columns=[
        "ID", "Name", "Available Pieces", "Current Price Per Piece", "Total Value", "Total Purchased", "Total Consumed"
    ]),
    "kits_bom": pd.DataFrame(columns=[
        "Kit ID", "Kit Name", "Material ID", "Material Name", "Qty Per Kit", "Unit Price Ref"
    ]),
    "created_kits": pd.DataFrame(columns=["Date", "Kit ID", "Kit Name", "Qty", "Notes"]),
    "sold_kits": pd.DataFrame(columns=[
        "Date", "Kit ID", "Kit Name", "Qty", "Platform", "Price", "Fees", "Amount Received",  "Notes"
    ]),
    "defective": pd.DataFrame(columns=["Date", "Material ID", "Qty", "Reason"]),
    "restock": pd.DataFrame(columns=["Material ID", "Desired Level", "Notes"]),
}

# ----------------------------
# Helpers: load/save/cache
# ----------------------------
@st.cache_data(show_spinner=False)
def load_df(path: str, kind: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return DEFAULTS[kind].copy()
    return DEFAULTS[kind].copy()

def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def refresh_cache():
    load_df.clear()  # type: ignore

# ----------------------------
# Domain helpers / inventory recompute
# ----------------------------
def ensure_numeric(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

def build_material_events(purchases: pd.DataFrame,
                          created_kits: pd.DataFrame,
                          defective: pd.DataFrame,
                          bom: pd.DataFrame) -> dict:
    """
    Build per-material chronological events from purchases (+pieces, cost) and consumptions (created kits, defective).
    Returns dict material_id -> list of events (date, delta_pieces, price_per_piece or None)
    Purchases: delta_pieces > 0, price_per_piece given.
    Consumptions: delta_pieces < 0, price_per_piece None (cost removed at current avg).
    """
    events_by_mat = {}

    # purchases
    for _, r in purchases.iterrows():
        try:
            d = pd.to_datetime(r["Date"], errors="coerce")
        except Exception:
            d = pd.to_datetime(datetime.today())
        mid = str(r["Material ID"])
        pieces = float(r.get("Pieces", 0) or 0)
        ppp = float(r.get("Price Per Piece", r.get("Price Per Piece", 0) or r.get("Price per Piece", 0) or 0))
        ev = {"date": d, "delta": pieces, "price_per_piece": ppp, "type": "purchase", "vendor": r.get("Vendor", ""), "name": r.get("Material Name", "")}
        events_by_mat.setdefault(mid, []).append(ev)

    # created_kits -> expand into material-level consumption events using BOM
    if (bom is not None) and (not bom.empty) and (created_kits is not None) and (not created_kits.empty):
        # prepare BOM per kit
        bom_map = {}
        for _, br in bom.iterrows():
            k = str(br["Kit ID"])
            bom_map.setdefault(k, []).append(br)
        for _, cr in created_kits.iterrows():
            try:
                d = pd.to_datetime(cr["Date"], errors="coerce")
            except Exception:
                d = pd.to_datetime(datetime.today())
            kit_id = str(cr["Kit ID"])
            qty_kits = float(cr.get("Qty", 0) or 0)
            if kit_id in bom_map:
                for br in bom_map[kit_id]:
                    mid = str(br["Material ID"])
                    qty_per_kit = float(br.get("Qty Per Kit", 0) or 0)
                    total_consume = qty_per_kit * qty_kits
                    ev = {"date": d, "delta": -total_consume, "price_per_piece": None, "type": "consume", "kit_id": kit_id, "kit_name": cr.get("Kit Name", "")}
                    events_by_mat.setdefault(mid, []).append(ev)

    # defective -> consumption events
    if defective is not None and not defective.empty:
        for _, dr in defective.iterrows():
            try:
                d = pd.to_datetime(dr["Date"], errors="coerce")
            except Exception:
                d = pd.to_datetime(datetime.today())
            mid = str(dr["Material ID"])
            qty = float(dr.get("Qty", 0) or 0)
            ev = {"date": d, "delta": -qty, "price_per_piece": None, "type": "defect", "reason": dr.get("Reason", "")}
            events_by_mat.setdefault(mid, []).append(ev)

    # sort events by date for each material
    for mid, evs in events_by_mat.items():
        evs_sorted = sorted(evs, key=lambda x: pd.to_datetime(x["date"]))
        events_by_mat[mid] = evs_sorted

    return events_by_mat

def compute_snapshot_from_events(events_by_mat: dict) -> pd.DataFrame:
    """
    For every material id, process its events in chronological order with average-cost method.
    Returns a master snapshot DataFrame with ID, Name, Available Pieces, Current Price Per Piece, Total Value, Total Purchased, Total Consumed.
    """
    rows = []
    for mid, events in events_by_mat.items():
        total_pieces = 0.0
        total_cost = 0.0  # cost associated with current stock
        total_purchased = 0.0
        total_consumed = 0.0
        name = ""
        vendor = ""
        for ev in events:
            if ev["type"] == "purchase":
                pieces = float(ev["delta"])
                price = float(ev["price_per_piece"] or 0.0)
                # add pieces and cost
                total_pieces += pieces
                total_cost += pieces * price
                total_purchased += pieces
                # capture name/vendor (latest purchase info)
                if ev.get("name"):
                    name = ev.get("name")
                if ev.get("vendor"):
                    vendor = ev.get("vendor")
            else:
                # consumption (delta negative)
                consume_pieces = -float(ev["delta"])  # positive number consumed
                # if no pieces to consume, allow going to negative (but we will keep it as negative and continue)
                if total_pieces > 0:
                    avg_cost = total_cost / total_pieces if total_pieces != 0 else 0.0
                    # remove cost proportionally
                    remove_cost = min(consume_pieces, total_pieces) * avg_cost
                    total_cost -= remove_cost
                    total_pieces -= min(consume_pieces, total_pieces)
                else:
                    # no stock to deduct from — reduce total_pieces to negative and keep cost zero
                    total_pieces -= consume_pieces
                total_consumed += consume_pieces
        # final snapshot values
        available_pieces = round(total_pieces, 6)
        current_price = round((total_cost / total_pieces) if total_pieces > 0 else 0.0, 4)
        total_value = round(available_pieces * current_price, 2)
        rows.append({
            "ID": mid,
            "Name": name,
            "Vendor": vendor,
            "Available Pieces": available_pieces,
            "Current Price Per Piece": current_price,
            "Total Value": total_value,
            "Total Purchased": round(total_purchased, 6),
            "Total Consumed": round(total_consumed, 6)
        })
    if not rows:
        return DEFAULTS["master"].copy()
    df = pd.DataFrame(rows)
    # ensure consistent order
    df = df.sort_values("ID").reset_index(drop=True)
    return df

def recompute_master_snapshot():
    purchases = load_df(FILES["purchases"], "purchases")
    created = load_df(FILES["created_kits"], "created_kits")
    defective = load_df(FILES["defective"], "defective")
    bom = load_df(FILES["kits_bom"], "kits_bom")
    events = build_material_events(purchases, created, defective, bom)
    snapshot = compute_snapshot_from_events(events)
    save_df(snapshot, FILES["master"])
    return snapshot

def get_material_name(snapshot: pd.DataFrame, material_id: str) -> str:
    row = snapshot[snapshot["ID"] == material_id]
    return row["Name"].iat[0] if not row.empty else ""

def can_build_kits(snapshot: pd.DataFrame, bom: pd.DataFrame, kit_id: str, qty: int) -> Tuple[bool, List[str]]:
    if not kit_id or qty <= 0 or bom.empty:
        return False, ["Invalid kit or quantity."]
    # Check stock
    msgs = []
    ok = True
    kit_rows = bom[bom["Kit ID"] == kit_id]
    for _, r in kit_rows.iterrows():
        mid = r["Material ID"]
        req_qty = r.get("Qty Per Kit", 0) * qty
        stock_row = snapshot[snapshot["ID"] == mid]
        avail = float(stock_row["Available Pieces"].iat[0]) if not stock_row.empty else 0
        if avail < req_qty:
            msgs.append(f"{mid} ({r['Material Name']}): need {req_qty}, available {avail}")
            ok = False
    return ok, msgs


# ----------------------------
# Utility for kit cost using snapshot
# ----------------------------
def kit_cost_from_snapshot(snapshot: pd.DataFrame, bom: pd.DataFrame, kit_id: str) -> float:
    if bom.empty or snapshot.empty or not kit_id:
        return 0.0
    rows = bom[bom["Kit ID"] == kit_id]
    total = 0.0
    for _, r in rows.iterrows():
        mid = str(r["Material ID"])
        qty = float(r.get("Qty Per Kit", 0) or 0)
        # get price per piece from snapshot
        price_row = snapshot[snapshot["ID"] == mid]
        price = float(price_row["Current Price Per Piece"].iat[0]) if not price_row.empty else 0.0
        unit_override = r.get("Unit Price Ref", None)
        unit_override = float(unit_override) if pd.notna(unit_override) and unit_override != 0 else 0.0
        unit = unit_override if unit_override > 0 else price
        total += qty * unit
    return round(total, 2)

# ----------------------------
# Load data initially
# ----------------------------
purchases_df = load_df(FILES["purchases"], "purchases")
bom_df = load_df(FILES["kits_bom"], "kits_bom")
created_df = load_df(FILES["created_kits"], "created_kits")
sold_df = load_df(FILES["sold_kits"], "sold_kits")
defective_df = load_df(FILES["defective"], "defective")
restock_df = load_df(FILES["restock"], "restock")

# compute snapshot (master) from ledger
master_snapshot = recompute_master_snapshot()

# ensure numeric columns in sold_df
ensure_numeric(sold_df, ["Price", "Fees", "Amount Received", "Cost Price", "Profit", "Qty"])
if "Qty" not in sold_df.columns and not sold_df.empty:
    sold_df["Qty"] = 1

# ----------------------------
# Navigation
# ----------------------------
MAIN_SECTIONS = ["📊 Dashboards", "📦 Inventory Management", "🧩 Kits Management", "🧾 Sales", "⬇️ Data"]
main_section = st.sidebar.selectbox("Section", MAIN_SECTIONS)

# ----------------------------
# Dashboards
# ----------------------------
if main_section == "📊 Dashboards":
    dash_choice = st.sidebar.radio("Choose Dashboard", ["📦 Inventory Dashboard", "💰 Sales Dashboard"])

    if dash_choice == "📦 Inventory Dashboard":
        st.title("📦 Inventory Dashboard")
        # KPIs
        total_materials = len(master_snapshot)
        total_pieces = int(master_snapshot["Available Pieces"].sum()) if not master_snapshot.empty else 0
        inventory_value = float(master_snapshot["Total Value"].sum()) if not master_snapshot.empty else 0.0
        total_kit_types = bom_df["Kit ID"].nunique() if not bom_df.empty else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Materials", f"{total_materials}")
        c2.metric("Available Pieces", f"{total_pieces:,}")
        c3.metric("Inventory Value", f"₹{inventory_value:,.2f}")
        c4.metric("Kit Types", f"{total_kit_types}")

        st.markdown("---")
        st.subheader("Low Stock")
        threshold = st.number_input("Low stock threshold (pieces)", min_value=1, value=5)
        low_stock = master_snapshot[pd.to_numeric(master_snapshot["Available Pieces"], errors="coerce").fillna(0) <= threshold]
        with st.expander("View low-stock items"):
            st.dataframe(low_stock if not low_stock.empty else pd.DataFrame([{"Status":"✅ All good"}]))

        st.markdown("---")
        st.subheader("Inventory Snapshot")
        with st.expander("View full inventory snapshot"):
            st.dataframe(master_snapshot)

    elif dash_choice == "💰 Sales Dashboard":
        st.title("💰 Sales Dashboard")
        if sold_df.empty:
            st.info("No sales recorded yet.")
        else:
            sold_local = sold_df.copy()
            sold_local["Date_dt"] = pd.to_datetime(sold_local["Date"], errors="coerce")
            st.markdown("### 🔍 Filters")
            col1, col2, col3 = st.columns([2,2,2])
            min_date = sold_local["Date_dt"].min().date() if not sold_local["Date_dt"].isna().all() else date.today()
            max_date = sold_local["Date_dt"].max().date() if not sold_local["Date_dt"].isna().all() else date.today()
            with col1:
                dr = st.date_input("Date Range", value=(min_date, max_date))
            platforms = sorted(sold_local["Platform"].dropna().unique().tolist())
            with col2:
                pf = st.multiselect("Platforms", options=platforms, default=platforms)
            kits = sorted(sold_local["Kit Name"].dropna().unique().tolist())
            with col3:
                kit_filter = st.multiselect("Kits", options=kits, default=kits)

            mask = pd.Series(True, index=sold_local.index)
            if isinstance(dr, tuple) and len(dr) == 2:
                mask &= (sold_local["Date_dt"].dt.date >= dr[0]) & (sold_local["Date_dt"].dt.date <= dr[1])
            if pf:
                mask &= sold_local["Platform"].isin(pf)
            if kit_filter:
                mask &= sold_local["Kit Name"].isin(kit_filter)

            sold_filtered = sold_local.loc[mask]
            ensure_numeric(sold_filtered, ["Amount Received", "Cost Price", "Profit", "Qty"])

            rev = float(sold_filtered["Amount Received"].sum())
            cost = float(sold_filtered["Cost Price"].sum())
            prof = float(sold_filtered["Profit"].sum())
            qty = int(sold_filtered["Qty"].sum()) if not sold_filtered.empty else 0

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Revenue", f"₹{rev:,.2f}")
            d2.metric("Cost", f"₹{cost:,.2f}")
            d3.metric("Profit", f"₹{prof:,.2f}")
            d4.metric("Units Sold", f"{qty}")

            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["📊 Charts", "📦 Kits", "📋 Raw Data"])
            with tab1:
                st.subheader("Profit by Platform")
                if not sold_filtered.empty:
                    prof_pf = sold_filtered.groupby("Platform", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False)
                    st.bar_chart(prof_pf.set_index("Platform"))
                st.subheader("Profit Over Time (Weekly)")
                if not sold_filtered.empty:
                    pt = sold_filtered.groupby(pd.Grouper(key="Date_dt", freq="W"))["Profit"].sum().fillna(0)
                    st.line_chart(pt)
            with tab2:
                st.subheader("Top-selling Kits (by Qty)")
                if not sold_filtered.empty:
                    top_k = sold_filtered.groupby("Kit Name", as_index=False)["Qty"].sum().sort_values("Qty", ascending=False).head(20)
                    st.table(top_k)
            with tab3:
                st.subheader("Filtered Sales Table")
                st.dataframe(sold_filtered.sort_values("Date_dt", ascending=False))

# ----------------------------
# Inventory Management
# ----------------------------
elif main_section == "📦 Inventory Management":
    st.title("📦 Inventory Management")
    sub = st.sidebar.radio("Inventory", ["Master Inventory (Purchases)", "Restock Planner", "Defective Items", "Purchase History"])

    if sub == "Master Inventory (Purchases)":
        st.subheader("Add Purchase (new or existing material)")
        # dropdown for existing materials (blank => add new)
        mat_options = [""] + sorted(purchases_df["Material ID"].astype(str).unique().tolist() + master_snapshot["ID"].astype(str).unique().tolist())
        mat_options = list(dict.fromkeys(mat_options))  # unique preserve order
        selected = st.selectbox("Select existing Material ID (leave blank to add new)", mat_options)

        # prefill values if selected from purchases or snapshot
        if selected:
            # attempt to prefill from latest purchase if exists else snapshot
            prev = purchases_df[purchases_df["Material ID"] == selected]
            if not prev.empty:
                last = prev.sort_values("Date").iloc[-1]
                default_name = last.get("Material Name", "")
                default_vendor = last.get("Vendor", "")
            else:
                row = master_snapshot[master_snapshot["ID"] == selected]
                default_name = row["Name"].iat[0] if not row.empty else ""
                default_vendor = row["Vendor"].iat[0] if not row.empty and "Vendor" in row.columns else ""
            col1, col2 = st.columns(2)
            with col1:
                m_name = st.text_input("Material Name", value=default_name)
                vendor = st.text_input("Vendor", value=default_vendor)
                date_p = st.date_input("Purchase Date", value=date.today())
            with col2:
                packs = st.number_input("Packs bought", min_value=0, step=1, value=0)
                qty_per_pack = st.number_input("Qty per Pack", min_value=0, step=1, value=0)
                cost_per_pack = st.number_input("Cost per Pack (₹)", min_value=0.0, step=0.01, value=0.0)
            pieces = int(packs * qty_per_pack)
            price_per_piece = float((cost_per_pack / qty_per_pack) if qty_per_pack > 0 else 0.0)
            st.markdown(f"**Pieces (computed):** {pieces} — **Price per piece:** ₹{price_per_piece:.4f}")
            if st.button("Add Purchase (update existing material)"):
                if not selected:
                    st.error("Select an existing ID or enter a new one.")
                else:
                    row = {
                        "Date": date_p.strftime("%Y-%m-%d"),
                        "Material ID": selected,
                        "Material Name": m_name.strip(),
                        "Vendor": vendor.strip(),
                        "Packs": packs,
                        "Qty Per Pack": qty_per_pack,
                        "Cost Per Pack": cost_per_pack,
                        "Pieces": pieces,
                        "Price Per Piece": price_per_piece
                    }
                    purchases_df = pd.concat([purchases_df, pd.DataFrame([row])], ignore_index=True)
                    save_df(purchases_df, FILES["purchases"])
                    # recompute snapshot
                    master_snapshot = recompute_master_snapshot()
                    refresh_cache()
                    st.success(f"Purchase added for {selected}: +{pieces} pcs at ₹{price_per_piece:.4f}/pc")

        else:
            st.markdown("**Add brand new material purchase**")
            col1, col2 = st.columns(2)
            with col1:
                new_id = st.text_input("New Material ID (unique)")
                m_name = st.text_input("Material Name")
                vendor = st.text_input("Vendor")
                date_p = st.date_input("Purchase Date", value=date.today())
            with col2:
                packs = st.number_input("Packs bought", min_value=0, step=1, value=0)
                qty_per_pack = st.number_input("Qty per Pack", min_value=0, step=1, value=0)
                cost_per_pack = st.number_input("Cost per Pack (₹)", min_value=0.0, step=0.01, value=0.0)
            pieces = int(packs * qty_per_pack)
            price_per_piece = float((cost_per_pack / qty_per_pack) if qty_per_pack > 0 else 0.0)
            st.markdown(f"**Pieces (computed):** {pieces} — **Price per piece:** ₹{price_per_piece:.4f}")
            if st.button("Add Purchase (new material)"):
                if not new_id:
                    st.error("Material ID required.")
                else:
                    if ((purchases_df["Material ID"] == new_id).any()) or ((master_snapshot["ID"] == new_id).any()):
                        st.error("Material ID already exists. Choose it from dropdown to update instead.")
                    else:
                        row = {
                            "Date": date_p.strftime("%Y-%m-%d"),
                            "Material ID": new_id.strip(),
                            "Material Name": m_name.strip(),
                            "Vendor": vendor.strip(),
                            "Packs": packs,
                            "Qty Per Pack": qty_per_pack,
                            "Cost Per Pack": cost_per_pack,
                            "Pieces": pieces,
                            "Price Per Piece": price_per_piece
                        }
                        purchases_df = pd.concat([purchases_df, pd.DataFrame([row])], ignore_index=True)
                        save_df(purchases_df, FILES["purchases"])
                        master_snapshot = recompute_master_snapshot()
                        refresh_cache()
                        st.success(f"Added new material {new_id} with {pieces} pcs at ₹{price_per_piece:.4f}/pc")

        st.markdown("---")
        st.subheader("Inventory Snapshot (derived)")
        with st.expander("View master snapshot"):
            st.dataframe(master_snapshot)

    elif sub == "Restock Planner":
        st.subheader("Restock Planner")
        threshold = st.number_input("Show items with stock <= (pieces)", min_value=1, value=5)
        low = master_snapshot[pd.to_numeric(master_snapshot["Available Pieces"], errors="coerce").fillna(0) <= threshold]
        st.dataframe(low[["ID", "Name", "Available Pieces", "Current Price Per Piece"]] if not low.empty else pd.DataFrame([{"Status":"✅ All good"}]))

        with st.form("restock_form"):
            rid = st.selectbox("Material ID", [""] + sorted(master_snapshot["ID"].astype(str).tolist()))
            desired = st.number_input("Desired Level (pieces)", min_value=0, step=1, value=0)
            notes = st.text_input("Notes")
            submitted = st.form_submit_button("Add / Update Restock")
            if submitted:
                if not rid:
                    st.error("Choose material")
                else:
                    restock_df = restock_df[restock_df["Material ID"] != rid]
                    restock_df = pd.concat([restock_df, pd.DataFrame([{"Material ID": rid, "Desired Level": desired, "Notes": notes}])], ignore_index=True)
                    save_df(restock_df, FILES["restock"])
                    refresh_cache()
                    st.success("Restock plan saved.")
        st.subheader("Restock Records")
        st.dataframe(restock_df)

    elif sub == "Defective Items":
        st.subheader("Defective / Damaged Items (decreases available stock)")
        mid = st.selectbox("Material ID", [""] + sorted(master_snapshot["ID"].astype(str).tolist()))
        qty = st.number_input("Qty (pieces)", min_value=1, step=1, value=1)
        reason = st.text_input("Reason")
        date_d = st.date_input("Date", value=date.today())
        if st.button("Log Defect & Deduct Stock"):
            if not mid:
                st.error("Choose material.")
            else:
                row = {"Date": date_d.strftime("%Y-%m-%d"), "Material ID": mid, "Qty": qty, "Reason": reason}
                defective_df = pd.concat([defective_df, pd.DataFrame([row])], ignore_index=True)
                save_df(defective_df, FILES["defective"])
                master_snapshot = recompute_master_snapshot()
                refresh_cache()
                st.success("Logged defective item and updated snapshot.")
        st.subheader("Defective Log")
        st.dataframe(defective_df)

    elif sub == "Purchase History":
        st.subheader("Purchase Ledger")
        filter_mat = st.selectbox("Filter by Material ID (optional)", [""] + sorted(purchases_df["Material ID"].astype(str).unique().tolist()))
        view = purchases_df.copy()
        if filter_mat:
            view = view[view["Material ID"] == filter_mat]
        view = view.sort_values("Date", ascending=False)
        st.dataframe(view)

# ----------------------------
# Kits Management
# ----------------------------
elif main_section == "🧩 Kits Management":
    st.title("🧩 Kits Management")
    sub = st.sidebar.radio("Kits", ["BOM (Kit components)", "Create Kits", "Kits Inventory"])

    if sub == "BOM (Kit components)":
        st.subheader("Define Kit (Kit ID + Kit Name) and add BOM rows")
        kit_id_input = st.text_input("Kit ID (e.g., KIT001)")
        kit_name_input = st.text_input("Kit Name (human readable)")
        st.markdown("**Add component to BOM**")
        col1, col2, col3 = st.columns(3)
        with col1:
            material_id = st.selectbox("Material ID", [""] + sorted(master_snapshot["ID"].astype(str).tolist()))
        with col2:
            qty_per = st.number_input("Qty Per Kit (pieces)", min_value=0.0, step=1.0, value=1.0)
        with col3:
            unit_override = st.number_input("Unit Price Override (₹) (optional)", min_value=0.0, step=0.01, value=0.0)
        if st.button("Add BOM Row"):
            if not kit_id_input or not kit_name_input or not material_id or qty_per <= 0:
                st.error("Provide Kit ID, Kit Name, Material, and qty > 0.")
            else:
                row = {
                    "Kit ID": kit_id_input.strip(),
                    "Kit Name": kit_name_input.strip(),
                    "Material ID": material_id,
                    "Material Name": get_material_name(master_snapshot if not master_snapshot.empty else DEFAULTS["master"], material_id),
                    "Qty Per Kit": qty_per,
                    "Unit Price Ref": unit_override if unit_override > 0 else None
                }
                bom_df = pd.concat([bom_df, pd.DataFrame([row])], ignore_index=True)
                save_df(bom_df, FILES["kits_bom"])
                refresh_cache()
                st.success(f"Added BOM row for {kit_id_input} ({kit_name_input}).")
        st.markdown("---")
        st.subheader("Current BOM")
        st.dataframe(bom_df)

    elif sub == "Create Kits":
        st.subheader("Produce / Assemble Kits (consume raw materials)")
        kit_options = (sorted(bom_df["Kit ID"].unique().tolist()) if not bom_df.empty else [])
        k_id = st.selectbox("Kit ID", [""] + kit_options)
        kit_name = ""
        if k_id:
            tmp = bom_df[bom_df["Kit ID"] == k_id]
            if not tmp.empty:
                kit_name = tmp["Kit Name"].iat[0]
        qty_build = st.number_input("Qty to create", min_value=1, step=1, value=1)
        notes = st.text_input("Notes (optional)")
        if st.button("Check Feasibility"):
            ok, msgs = can_build_kits(master_snapshot, bom_df, k_id, qty_build)
            if ok:
                st.success("Enough materials available ✅")
            else:
                st.error("Insufficient materials:")
                for m in msgs:
                    st.write("- ", m)
        if st.button("Produce Kits (Consume Raw Materials)"):
            ok, msgs = can_build_kits(master_snapshot, bom_df, k_id, qty_build)
            if not ok:
                st.error("Cannot create kits due to shortages.")
            else:
                # record created kit
                created_row = {"Date": datetime.today().strftime("%Y-%m-%d"), "Kit ID": k_id, "Kit Name": kit_name, "Qty": qty_build, "Notes": notes}
                created_df = pd.concat([created_df, pd.DataFrame([created_row])], ignore_index=True)
                save_df(created_df, FILES["created_kits"])
                # recompute snapshot (consumption will be applied when recomputing)
                master_snapshot = recompute_master_snapshot()
                refresh_cache()
                st.success(f"Created {qty_build} x {k_id} ({kit_name}).")
        st.subheader("Production Log (recent)")
        st.dataframe(created_df.sort_values("Date", ascending=False).head(200))

    elif sub == "Kits Inventory":
        st.subheader("Kits Inventory (Produced - Sold)")
        inv_df = pd.DataFrame()
        if not created_df.empty:
            inv_df = created_df.groupby("Kit ID", as_index=False)["Qty"].sum().rename(columns={"Qty": "Created"})
        sold_agg = pd.DataFrame()
        if not sold_df.empty:
            sold_agg = sold_df.groupby("Kit ID", as_index=False)["Qty"].sum().rename(columns={"Qty": "Sold"})
        inv = inv_df.merge(sold_agg, on="Kit ID", how="left").fillna({"Sold": 0})
        if not inv.empty:
            inv["Available"] = inv["Created"] - inv["Sold"]
            # add kit name
            kit_names = created_df[["Kit ID", "Kit Name"]].drop_duplicates("Kit ID")
            bom_names = bom_df[["Kit ID", "Kit Name"]].drop_duplicates("Kit ID")
            names = pd.concat([kit_names, bom_names]).drop_duplicates("Kit ID", keep="first")
            inv = inv.merge(names, on="Kit ID", how="left")
        st.dataframe(inv)

# ----------------------------
# Sales
# ----------------------------
elif main_section == "🧾 Sales":
    st.title("🧾 Sales")
    sub = st.sidebar.radio("Sales", ["Record Sale", "Sales Ledger"])

    if sub == "Record Sale":
        st.subheader("Record a sale (kit cost uses current weighted avg per material)")
        kit_list = sorted(bom_df["Kit ID"].unique().tolist()) if not bom_df.empty else []
        k_id = st.selectbox("Kit ID", [""] + kit_list)
        kit_name = ""
        if k_id:
            tmp = bom_df[bom_df["Kit ID"] == k_id]
            if not tmp.empty:
                kit_name = tmp["Kit Name"].iat[0]
        platform = st.selectbox("Platform", ["Amazon", "Meesho", "Flipkart", "Offline", "Other"])
        qty = st.number_input("Quantity sold", min_value=1, step=1, value=1)
        price = st.number_input("Sale price per kit (₹)", min_value=0.0, step=0.01, value=0.0)
        fees = st.number_input("Fees / commission (total ₹)", min_value=0.0, step=0.01, value=0.0)
        amount_received = st.number_input("Amount received (total ₹)", min_value=0.0, step=0.01, value=float(price * qty))
        notes = st.text_input("Notes (optional)")
        sale_date = st.date_input("Sale date", value=date.today())

        if st.button("Save Sale"):
            if not k_id:
                st.error("Choose a Kit ID.")
            else:
                # cost uses current snapshot
                cost_per_kit = kit_cost_from_snapshot(master_snapshot, bom_df, k_id)
                cost_total = cost_per_kit * int(qty)
                profit = round(float(amount_received) - float(cost_total), 2)
                row = {
                    "Date": sale_date.strftime("%Y-%m-%d"),
                    "Kit ID": k_id,
                    "Kit Name": kit_name,
                    "Qty": int(qty),
                    "Platform": platform,
                    "Price": float(price),
                    "Fees": float(fees),
                    "Amount Received": float(amount_received),
                    "Cost Price": float(cost_total),
                    "Profit": float(profit),
                    "Notes": notes
                }
                sold_df = pd.concat([sold_df, pd.DataFrame([row])], ignore_index=True)
                save_df(sold_df, FILES["sold_kits"])
                # sales don't change inventory directly; inventory consumption is recorded when kits are produced (created_kits)
                refresh_cache()
                st.success(f"Recorded sale. Profit: ₹{profit:,.2f}")

        st.subheader("Recent Sales")
        st.dataframe(sold_df.sort_values("Date", ascending=False).head(200))

    elif sub == "Sales Ledger":
        st.subheader("Sales Ledger")
        st.dataframe(sold_df)

# ----------------------------
# Data (Export / Templates / Import)
# ----------------------------
elif main_section == "⬇️ Data":
    st.title("⬇️ Import / ⬆️ Export Data & Templates")
    st.markdown("Download current datasets or download header-only templates. Upload CSV to replace dataset (must match template columns).")

    # Exports
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("Download purchases.csv", purchases_df.to_csv(index=False).encode("utf-8"), "purchases.csv")
        st.download_button("Download master_snapshot.csv", master_snapshot.to_csv(index=False).encode("utf-8"), "master_snapshot.csv")
    with e2:
        st.download_button("Download kits_bom.csv", bom_df.to_csv(index=False).encode("utf-8"), "kits_bom.csv")
        st.download_button("Download created_kits.csv", created_df.to_csv(index=False).encode("utf-8"), "created_kits.csv")
    with e3:
        st.download_button("Download sold_kits.csv", sold_df.to_csv(index=False).encode("utf-8"), "sold_kits.csv")
        st.download_button("Download defective_items.csv", defective_df.to_csv(index=False).encode("utf-8"), "defective_items.csv")

    st.markdown("---")
    st.subheader("📄 Download CSV templates (headers only)")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.download_button("Template: purchases.csv", DEFAULTS["purchases"].to_csv(index=False).encode("utf-8"), "template_purchases.csv")
        st.download_button("Template: master_snapshot.csv", DEFAULTS["master"].to_csv(index=False).encode("utf-8"), "template_master_snapshot.csv")
    with t2:
        st.download_button("Template: kits_bom.csv", DEFAULTS["kits_bom"].to_csv(index=False).encode("utf-8"), "template_kits_bom.csv")
        st.download_button("Template: created_kits.csv", DEFAULTS["created_kits"].to_csv(index=False).encode("utf-8"), "template_created_kits.csv")
    with t3:
        st.download_button("Template: sold_kits.csv", DEFAULTS["sold_kits"].to_csv(index=False).encode("utf-8"), "template_sold_kits.csv")
        st.download_button("Template: defective.csv", DEFAULTS["defective"].to_csv(index=False).encode("utf-8"), "template_defective.csv")

    st.markdown("---")
    st.subheader("Upload CSV to replace a dataset (must match template columns)")
    which_map = {
        "purchases": ("purchases", FILES["purchases"]),
        "kits_bom": ("kits_bom", FILES["kits_bom"]),
        "created_kits": ("created_kits", FILES["created_kits"]),
        "sold_kits": ("sold_kits", FILES["sold_kits"]),
        "defective": ("defective", FILES["defective"]),
        "restock": ("restock", FILES["restock"]),
    }
    which_choice = st.selectbox("Choose dataset to replace", list(which_map.keys()))
    up = st.file_uploader(f"Upload CSV for {which_choice}", type=["csv"], key="up_"+which_choice)
    if up is not None and st.button("Replace dataset"):
        try:
            new_df = pd.read_csv(up)
            expected = DEFAULTS[which_map[which_choice][0]].columns.tolist()
            if list(new_df.columns) != expected:
                st.error(f"Invalid columns. Expected headers: {expected}")
            else:
                save_df(new_df, which_map[which_choice][1])
                # If we replaced purchases/created/defective, recompute master snapshot
                if which_choice in ("purchases", "created_kits", "defective", "kits_bom"):
                    master_snapshot = recompute_master_snapshot()
                refresh_cache()
                st.success(f"Replaced {which_choice} with uploaded CSV ({len(new_df)} rows).")
        except Exception as e:
            st.error(f"Failed to import: {e}")

# ----------------------------
# Footer
# ----------------------------
st.sidebar.markdown("---")
if st.sidebar.button("Refresh data (clear cache)"):
    refresh_cache()
    st.experimental_rerun()
