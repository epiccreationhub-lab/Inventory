# inventory_app.py
"""
Inventory & Kits Manager (Google Sheets backed, batched reads + retries)

Place your service account JSON in Streamlit secrets:
st.secrets["gcp_service_account"] = { ... }

Spreadsheet: BusinessData
Expected tabs (worksheets): purchases, master, kits_bom, created_kits, sold_kits, defective, restock
"""
import time
import random
from datetime import datetime, date
from typing import List, Tuple, Optional, Dict

import streamlit as st
import pandas as pd

# Google Sheets / OAuth
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except Exception as e:
    st.error("Missing dependencies: install gspread and oauth2client in the environment.")
    raise

st.set_page_config(page_title="Inventory & Kits Manager", page_icon="📦", layout="wide")

# ---------- Config ----------
SPREADSHEET_NAME = "BusinessData"
CACHE_TTL = int(st.secrets.get("gsheet_cache_ttl", 300))  # seconds, default 300
_MAX_RETRIES = 5
_BACKOFF_BASE = 1.0  # seconds
_READ_RANGE_SUFFIX = "A:Z"  # Range to fetch per sheet; adjust if more columns required

EXPECTED_TABS = [
    "purchases",
    "master",
    "kits_bom",
    "created_kits",
    "sold_kits",
    "defective",
    "restock",
]

# ---------- Defaults / Templates ----------
DEFAULTS = {
    "purchases": pd.DataFrame(columns=[
        "Date", "Material ID", "Material Name", "Vendor", "Packs", "Qty Per Pack",
        "Cost Per Pack", "Pieces", "Price Per Piece"
    ]),
    "master": pd.DataFrame(columns=[
        "ID", "Name", "Available Pieces", "Current Price Per Piece",
        "Total Value", "Total Purchased", "Total Consumed"
    ]),
    "kits_bom": pd.DataFrame(columns=[
        "Kit ID", "Kit Name", "Material ID", "Material Name", "Qty Per Kit", "Unit Price Ref"
    ]),
    "created_kits": pd.DataFrame(columns=["Date", "Kit ID", "Kit Name", "Qty", "Notes"]),
    "sold_kits": pd.DataFrame(columns=[
        "Date", "Kit ID", "Kit Name", "Qty", "Platform", "Price", "Fees",
        "Amount Received", "Cost Price", "Profit", "Notes"
    ]),
    "defective": pd.DataFrame(columns=["Date", "Material ID", "Qty", "Reason"]),
    "restock": pd.DataFrame(columns=["Material ID", "Desired Level", "Notes"]),
}

# ---------- Authorize with Google Sheets ----------
if "gcp_service_account" not in st.secrets:
    st.error("Missing `gcp_service_account` in Streamlit secrets — add your service account JSON there.")
    st.stop()

creds_dict = st.secrets["gcp_service_account"]
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]
try:
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    gclient = gspread.authorize(creds)
except Exception as exc:
    st.error(f"Failed to authorize Google Sheets: {exc}")
    raise

# ---------- Helper utilities ----------
def _sheet_range(tab_name: str) -> str:
    return f"{tab_name}!{_READ_RANGE_SUFFIX}"

def _safe_float(val) -> float:
    try:
        if val in (None, ""):
            return 0.0
        return float(val)
    except Exception:
        return 0.0

def ensure_numeric(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# ---------- Batched loader with retries & caching ----------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def gsheet_load_all(tabs: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Batch-load multiple worksheets using one values_batch_get call.
    Returns a dict: tab_name -> DataFrame
    Cached for CACHE_TTL seconds.
    """
    sh = gclient.open(SPREADSHEET_NAME)
    ranges = [_sheet_range(t) for t in tabs]

    # Attempt batched call with retries
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = sh.values_batch_get(ranges)
            outputs: Dict[str, pd.DataFrame] = {}
            value_ranges = result.get("valueRanges", [])
            for idx, vr in enumerate(value_ranges):
                tab = tabs[idx]
                values = vr.get("values", []) or []
                if not values:
                    outputs[tab] = DEFAULTS.get(tab, pd.DataFrame()).copy()
                    continue
                header = values[0]
                rows = values[1:]
                # Build dataframe; if rows length mismatches, pad/truncate automatically handled by pandas
                df = pd.DataFrame(rows, columns=header)
                # If expected header set matches, reorder to canonical order
                expected = list(DEFAULTS.get(tab, pd.DataFrame()).columns)
                if expected and set(df.columns) == set(expected):
                    df = df[expected]
                outputs[tab] = df
            return outputs
        except Exception as exc:
            estr = str(exc)
            # detect rate-limit / quota-like errors heuristically
            if "429" in estr or "Quota" in estr or "rateLimitExceeded" in estr or "userRateLimitExceeded" in estr:
                wait = _BACKOFF_BASE * (2 ** (attempt - 1))
                # small jitter to spread retries
                wait = wait + random.uniform(0, 0.2 * attempt)
                st.warning(f"Sheets API rate-limited (attempt {attempt}/{_MAX_RETRIES}). Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            # other errors: break to fallback
            st.warning(f"Batch load failed: {exc}. Falling back to individual sheet loads.")
            break

    # Fallback: load per sheet with retries
    outputs = {}
    for tab in tabs:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                ws = sh.worksheet(tab)
                data = ws.get_all_records()
                df = pd.DataFrame(data)
                if df.empty:
                    outputs[tab] = DEFAULTS.get(tab, pd.DataFrame()).copy()
                else:
                    expected = list(DEFAULTS.get(tab, pd.DataFrame()).columns)
                    if expected and set(df.columns) == set(expected):
                        df = df[expected]
                    outputs[tab] = df
                break
            except Exception as exc:
                estr = str(exc)
                if "429" in estr or "Quota" in estr or "rateLimitExceeded" in estr or "userRateLimitExceeded" in estr:
                    wait = _BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 0.1 * attempt)
                    st.warning(f"Sheet {tab} read rate-limited (attempt {attempt}). Retrying in {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                st.warning(f"Failed to load sheet {tab}: {exc}. Using empty template.")
                outputs[tab] = DEFAULTS.get(tab, pd.DataFrame()).copy()
                break
    return outputs

# ---------- Save helper (per sheet) and clear batch cache ----------
def gsheet_save_single(tab_name: str, df: pd.DataFrame):
    """
    Overwrite one worksheet with the dataframe contents.
    Clears the cached batch loader afterwards.
    """
    sh = gclient.open(SPREADSHEET_NAME)
    try:
        try:
            ws = sh.worksheet(tab_name)
        except gspread.WorksheetNotFound:
            # Create worksheet with reasonable size
            ws = sh.add_worksheet(title=tab_name, rows="1000", cols="30")
        cols = df.columns.tolist()
        rows = df.fillna("").values.tolist()
        ws.clear()
        # Use update to write header + rows
        ws.update([cols] + rows)
    except Exception as exc:
        st.warning(f"Failed to save sheet '{tab_name}': {exc}")
        raise
    finally:
        # Clear the batch cache so subsequent reads see the latest data
        try:
            gsheet_load_all.clear()
        except Exception:
            pass

# ---------- IO wrappers using tab keys ----------
def load_df(kind: str) -> pd.DataFrame:
    if kind not in EXPECTED_TABS:
        raise ValueError(f"Unknown kind: {kind}")
    loaded = gsheet_load_all(EXPECTED_TABS)
    return loaded.get(kind, DEFAULTS.get(kind).copy())

def save_df(df: pd.DataFrame, kind: str):
    if kind not in EXPECTED_TABS:
        raise ValueError(f"Unknown kind: {kind}")
    # reorder to canonical columns if matches
    expected = list(DEFAULTS.get(kind).columns)
    if expected and set(df.columns) == set(expected):
        df = df[expected]
    gsheet_save_single(kind, df)

# ---------- Domain logic (robust) ----------
def build_material_events(purchases: pd.DataFrame,
                          created_kits: pd.DataFrame,
                          defective: pd.DataFrame,
                          bom: pd.DataFrame) -> dict:
    events_by_mat = {}
    # purchases
    if purchases is not None and not purchases.empty:
        for r in purchases.to_dict(orient="records"):
            try:
                d = pd.to_datetime(r.get("Date"), errors="coerce")
            except Exception:
                d = pd.to_datetime(datetime.today())
            mid = str(r.get("Material ID") or "")
            pieces = _safe_float(r.get("Pieces") or r.get("pieces") or 0)
            if pieces == 0:
                packs = _safe_float(r.get("Packs") or 0)
                qty_per_pack = _safe_float(r.get("Qty Per Pack") or 0)
                pieces = packs * qty_per_pack
            # price candidates
            price_candidates = [
                r.get("Price Per Piece"), r.get("Price per Piece"),
                r.get("PricePerPiece"), r.get("Price_Per_Piece"),
                r.get("Price"), r.get("Cost Per Pack")
            ]
            ppp = next((x for x in price_candidates if x not in (None, "") and pd.notna(x)), None)
            ppp = _safe_float(ppp)
            if ppp == 0 and _safe_float(r.get("Cost Per Pack") or 0) > 0 and _safe_float(r.get("Qty Per Pack") or 0) > 0:
                ppp = _safe_float(r.get("Cost Per Pack")) / _safe_float(r.get("Qty Per Pack"))
            ev = {"date": d, "delta": pieces, "price_per_piece": ppp, "type": "purchase",
                  "vendor": r.get("Vendor", ""), "name": r.get("Material Name", "")}
            events_by_mat.setdefault(mid, []).append(ev)
    # created kits -> expand using bom
    if bom is not None and not bom.empty and created_kits is not None and not created_kits.empty:
        bom_map = {}
        for br in bom.to_dict(orient="records"):
            k = str(br.get("Kit ID") or "")
            bom_map.setdefault(k, []).append(br)
        for cr in created_kits.to_dict(orient="records"):
            try:
                d = pd.to_datetime(cr.get("Date"), errors="coerce")
            except Exception:
                d = pd.to_datetime(datetime.today())
            kit_id = str(cr.get("Kit ID") or "")
            qty_kits = _safe_float(cr.get("Qty") or 0)
            if kit_id in bom_map:
                for br in bom_map[kit_id]:
                    mid = str(br.get("Material ID") or "")
                    qty_per_kit = _safe_float(br.get("Qty Per Kit") or 0)
                    total_consume = qty_per_kit * qty_kits
                    ev = {"date": d, "delta": -total_consume, "price_per_piece": None,
                          "type": "consume", "kit_id": kit_id, "kit_name": cr.get("Kit Name", "")}
                    events_by_mat.setdefault(mid, []).append(ev)
    # defective
    if defective is not None and not defective.empty:
        for dr in defective.to_dict(orient="records"):
            try:
                d = pd.to_datetime(dr.get("Date"), errors="coerce")
            except Exception:
                d = pd.to_datetime(datetime.today())
            mid = str(dr.get("Material ID") or "")
            qty = _safe_float(dr.get("Qty") or 0)
            ev = {"date": d, "delta": -qty, "price_per_piece": None, "type": "defect", "reason": dr.get("Reason", "")}
            events_by_mat.setdefault(mid, []).append(ev)
    # sort
    for mid, evs in events_by_mat.items():
        evs_sorted = sorted(evs, key=lambda x: pd.to_datetime(x.get("date")))
        events_by_mat[mid] = evs_sorted
    return events_by_mat

def compute_snapshot_from_events(events_by_mat: dict) -> pd.DataFrame:
    rows = []
    for mid, events in events_by_mat.items():
        total_pieces = 0.0
        total_cost = 0.0
        total_purchased = 0.0
        total_consumed = 0.0
        name = ""
        vendor = ""
        for ev in events:
            if ev.get("type") == "purchase":
                pieces = _safe_float(ev.get("delta") or 0)
                price = _safe_float(ev.get("price_per_piece") or 0)
                total_pieces += pieces
                total_cost += pieces * price
                total_purchased += pieces
                if ev.get("name"):
                    name = ev.get("name")
                if ev.get("vendor"):
                    vendor = ev.get("vendor")
            else:
                consume_pieces = -_safe_float(ev.get("delta") or 0)
                if total_pieces > 0:
                    avg_cost = (total_cost / total_pieces) if total_pieces != 0 else 0.0
                    remove_cost = min(consume_pieces, total_pieces) * avg_cost
                    total_cost -= remove_cost
                    total_pieces -= min(consume_pieces, total_pieces)
                else:
                    total_pieces -= consume_pieces
                total_consumed += consume_pieces
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
    df = df.sort_values("ID").reset_index(drop=True)
    return df

def recompute_master_snapshot() -> pd.DataFrame:
    # Clear batch cache
    try:
        gsheet_load_all.clear()
    except Exception:
        pass
    purchases = load_df("purchases")
    created = load_df("created_kits")
    defective = load_df("defective")
    bom = load_df("kits_bom")
    events = build_material_events(purchases, created, defective, bom)
    snapshot = compute_snapshot_from_events(events)
    save_df(snapshot, "master")
    return snapshot

def get_material_name(snapshot: pd.DataFrame, material_id: str) -> str:
    row = snapshot[snapshot["ID"] == material_id]
    return row["Name"].iat[0] if not row.empty else ""

def can_build_kits(snapshot: pd.DataFrame, bom: pd.DataFrame, kit_id: str, qty: int) -> Tuple[bool, List[str]]:
    if not kit_id or qty <= 0:
        return False, ["Invalid kit or quantity."]
    if bom is None or bom.empty:
        return False, ["BOM is empty or not defined."]
    kit_rows = bom[bom["Kit ID"] == kit_id]
    if kit_rows.empty:
        return False, [f"Kit {kit_id} not defined in BOM."]
    msgs = []
    ok = True
    for _, r in kit_rows.iterrows():
        mid = r["Material ID"]
        req_qty = _safe_float(r.get("Qty Per Kit", 0)) * qty
        stock_row = snapshot[snapshot["ID"] == mid]
        avail = float(stock_row["Available Pieces"].iat[0]) if not stock_row.empty else 0
        if avail < req_qty:
            msgs.append(f"{mid} ({r.get('Material Name','')}): need {req_qty}, available {avail}")
            ok = False
    return ok, msgs

def kit_cost_from_snapshot(snapshot: pd.DataFrame, bom: pd.DataFrame, kit_id: str) -> float:
    if bom is None or bom.empty or snapshot is None or snapshot.empty or not kit_id:
        return 0.0
    rows = bom[bom["Kit ID"] == kit_id]
    total = 0.0
    for _, r in rows.iterrows():
        mid = str(r["Material ID"])
        qty = _safe_float(r.get("Qty Per Kit", 0) or 0)
        price_row = snapshot[snapshot["ID"] == mid]
        price = float(price_row["Current Price Per Piece"].iat[0]) if not price_row.empty else 0.0
        unit_override = r.get("Unit Price Ref", None)
        unit_override = _safe_float(unit_override) if unit_override not in (None, "") and pd.notna(unit_override) else 0.0
        unit = unit_override if unit_override > 0 else price
        total += qty * unit
    return round(total, 2)

# ---------- Load initial data ----------
loaded = gsheet_load_all(EXPECTED_TABS)
purchases_df = loaded.get("purchases", DEFAULTS["purchases"].copy())
bom_df = loaded.get("kits_bom", DEFAULTS["kits_bom"].copy())
created_df = loaded.get("created_kits", DEFAULTS["created_kits"].copy())
sold_df = loaded.get("sold_kits", DEFAULTS["sold_kits"].copy())
defective_df = loaded.get("defective", DEFAULTS["defective"].copy())
restock_df = loaded.get("restock", DEFAULTS["restock"].copy())

master_snapshot = recompute_master_snapshot()

ensure_numeric(sold_df, ["Price", "Fees", "Amount Received", "Cost Price", "Profit", "Qty"])
if "Qty" not in sold_df.columns:
    sold_df["Qty"] = 0

# ---------- UI (keeps same layout) ----------
MAIN_SECTIONS = ["📊 Dashboards", "📦 Inventory Management", "🧩 Kits Management", "🧾 Sales", "⬇️ Data"]
main_section = st.sidebar.selectbox("Section", MAIN_SECTIONS)

# -- Dashboards --
if main_section == "📊 Dashboards":
    dash_choice = st.sidebar.radio("Choose Dashboard", ["📦 Inventory Dashboard", "💰 Sales Dashboard"])
    if dash_choice == "📦 Inventory Dashboard":
        st.title("📦 Inventory Dashboard")
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

# ---------- Inventory Management ----------
elif main_section == "📦 Inventory Management":
    st.title("📦 Inventory Management")
    sub = st.sidebar.radio("Inventory", ["Master Inventory (Purchases)", "Restock Planner", "Defective Items", "Purchase History"])

    if sub == "Master Inventory (Purchases)":
        st.subheader("Add Purchase (new or existing material)")
        existing_ids = sorted(pd.unique(list(purchases_df.get("Material ID", pd.Series(dtype=str)).astype(str).tolist() + master_snapshot.get("ID", pd.Series(dtype=str)).astype(str).tolist())))
        mat_options = [""] + existing_ids
        mat_options = list(dict.fromkeys(mat_options))
        selected = st.selectbox("Select existing Material ID (leave blank to add new)", mat_options)

        if selected:
            prev = purchases_df[purchases_df["Material ID"] == selected] if "Material ID" in purchases_df.columns else pd.DataFrame()
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
                    save_df(purchases_df, "purchases")
                    master_snapshot = recompute_master_snapshot()
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
                    if ((purchases_df.get("Material ID", pd.Series(dtype=str)) == new_id).any()) or ((master_snapshot.get("ID", pd.Series(dtype=str)) == new_id).any()):
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
                        save_df(purchases_df, "purchases")
                        master_snapshot = recompute_master_snapshot()
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
                    restock_df = restock_df[restock_df["Material ID"] != rid] if not restock_df.empty else restock_df
                    restock_df = pd.concat([restock_df, pd.DataFrame([{"Material ID": rid, "Desired Level": desired, "Notes": notes}])], ignore_index=True)
                    save_df(restock_df, "restock")
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
                save_df(defective_df, "defective")
                master_snapshot = recompute_master_snapshot()
                st.success("Logged defective item and updated snapshot.")
        st.subheader("Defective Log")
        st.dataframe(defective_df)

    elif sub == "Purchase History":
        st.subheader("Purchase Ledger")
        filter_mat = st.selectbox("Filter by Material ID (optional)", [""] + sorted(purchases_df.get("Material ID", pd.Series(dtype=str)).astype(str).unique().tolist()))
        view = purchases_df.copy()
        if filter_mat:
            view = view[view["Material ID"] == filter_mat]
        view = view.sort_values("Date", ascending=False)
        st.dataframe(view)

# ---------- Kits Management ----------
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
                save_df(bom_df, "kits_bom")
                st.success(f"Added BOM row for {kit_id_input} ({kit_name_input}).")
        st.markdown("---")
        st.subheader("Current BOM")
        st.dataframe(bom_df)

    elif sub == "Create Kits":
        st.subheader("Produce / Assemble Kits (consume raw materials)")
        kit_options = (sorted(bom_df["Kit ID"].dropna().unique().tolist()) if not bom_df.empty else [])
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
                created_row = {"Date": datetime.today().strftime("%Y-%m-%d"), "Kit ID": k_id, "Kit Name": kit_name, "Qty": qty_build, "Notes": notes}
                created_df = pd.concat([created_df, pd.DataFrame([created_row])], ignore_index=True)
                save_df(created_df, "created_kits")
                master_snapshot = recompute_master_snapshot()
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
            kit_names = created_df[["Kit ID", "Kit Name"]].drop_duplicates("Kit ID")
            bom_names = bom_df[["Kit ID", "Kit Name"]].drop_duplicates("Kit ID")
            names = pd.concat([kit_names, bom_names]).drop_duplicates("Kit ID", keep="first")
            inv = inv.merge(names, on="Kit ID", how="left")
        st.dataframe(inv)

# ---------- Sales ----------
elif main_section == "🧾 Sales":
    st.title("🧾 Sales")
    sub = st.sidebar.radio("Sales", ["Record Sale", "Sales Ledger"])

    if sub == "Record Sale":
        st.subheader("Record a sale (kit cost uses current weighted avg per material)")
        kit_list = sorted(bom_df["Kit ID"].dropna().unique().tolist()) if not bom_df.empty else []
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
                save_df(sold_df, "sold_kits")
                st.success(f"Recorded sale. Profit: ₹{profit:,.2f}")

        st.subheader("Recent Sales")
        st.dataframe(sold_df.sort_values("Date", ascending=False).head(200))

    elif sub == "Sales Ledger":
        st.subheader("Sales Ledger")
        st.dataframe(sold_df)

# ---------- Data (export / import) ----------
elif main_section == "⬇️ Data":
    st.title("⬇️ Import / ⬆️ Export Data & Templates")
    st.markdown("Download current datasets or download header-only templates. Upload CSV to replace a worksheet (must match template columns).")

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
    st.subheader("Upload CSV to replace a worksheet (must match template columns exactly)")
    which_map = {
        "purchases": "purchases",
        "kits_bom": "kits_bom",
        "created_kits": "created_kits",
        "sold_kits": "sold_kits",
        "defective": "defective",
        "restock": "restock",
    }
    which_choice = st.selectbox("Choose dataset to replace", list(which_map.keys()))
    up = st.file_uploader(f"Upload CSV for {which_choice}", type=["csv"], key="up_"+which_choice)
    if up is not None and st.button("Replace worksheet"):
        try:
            new_df = pd.read_csv(up)
            expected = list(DEFAULTS[which_map[which_choice]].columns)
            if set(new_df.columns) != set(expected):
                st.error(f"Invalid columns. Expected headers (order not important): {expected}")
            else:
                new_df = new_df[expected]  # reorder
                save_df(new_df, which_map[which_choice])
                if which_choice in ("purchases", "created_kits", "defective", "kits_bom"):
                    master_snapshot = recompute_master_snapshot()
                st.success(f"Replaced worksheet {which_choice} with uploaded CSV ({len(new_df)} rows).")
        except Exception as e:
            st.error(f"Failed to import: {e}")

# ---------- Footer ----------
st.sidebar.markdown("---")
if st.sidebar.button("Refresh data (clear cache)"):
    try:
        gsheet_load_all.clear()
    except Exception:
        pass
    st.experimental_rerun()
