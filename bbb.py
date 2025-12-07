import streamlit as st
import pandas as pd
from datetime import datetime, time, timedelta
import math
import numpy as np

# =================================================================
# 1. CONFIGURATION AND ORGANIZATION-SPECIFIC RULES
# =================================================================

SHIFT_START = time(9, 15)
SHIFT_END = time(17, 45)

LATE_UNIT_MINUTES = 15

# Original Morning Half Day trigger: IN time >= 10:31
# HALF_DAY_START_TIME = time(10, 31)

# NEW Half Day trigger (10:16 to 13:30 is considered Half Day)
HALF_DAY_START_TIME_NEW = time(10, 16)
HALF_DAY_END_TIME_NEW = time(13, 30)

# Evening Half Day trigger: OUT time <= 16:45
EVENING_HALF_DAY_TIME = time(16, 45)

# 1 Hour Leave window: 16:46 to 17:44
ONE_HOUR_LEAVE_START_TIME = time(16, 46)
ONE_HOUR_LEAVE_END_TIME = time(17, 44)

# =================================================================
# 2. CORE CALCULATION FUNCTIONS
# =================================================================

def to_time_safe(s):
    """Safely converts string to time object."""
    s = str(s).strip()
    if not s or s in ["00:00", "nan", "NaT"]:
        return None
    try:
        return datetime.strptime(s, "%H:%M").time()
    except ValueError:
        try:
            dt_obj = pd.to_datetime(s, errors='coerce')
            if pd.notna(dt_obj):
                return dt_obj.time()
        except:
            pass
    return None

def calculate_late_status(in_time):
    """Calculates 15-min late units or morning Half Day Leave."""
    if in_time is None: return 0, 'Missing IN Time', 0

    today = datetime.today().date()
    shift_start_dt = datetime.combine(today, SHIFT_START)
    in_time_dt = datetime.combine(today, in_time)

    # --- NEW Half Day Leave Check (IN time between 10:16 and 13:30) ---
    half_day_start_dt_new = datetime.combine(today, HALF_DAY_START_TIME_NEW)
    half_day_end_dt_new = datetime.combine(today, HALF_DAY_END_TIME_NEW)

    if in_time_dt >= half_day_start_dt_new and in_time_dt <= half_day_end_dt_new:
        return 0, 'Half Day Leave (Morning IN)', 1
    # ------------------------------------------------------------------

    # Late Minutes calculation (only if not an early Half Day)
    late_minutes = (in_time_dt - shift_start_dt).total_seconds() / 60
    if late_minutes <= 0: return 0, 'On Time', 0

    # 15 Minute Late check (applies if IN time is after SHIFT_START but before the Half Day Window)
    late_units = math.ceil(late_minutes / LATE_UNIT_MINUTES)
    return late_units, f'{late_units} x 15 Min Late', 0


def calculate_early_out_status(out_time):
    """Classifies 1-hour leave or evening Half Day Leave."""
    if out_time is None: return 0, 'Missing OUT Time', 0

    today = datetime.today().date()
    shift_end_dt = datetime.combine(today, SHIFT_END)
    out_time_dt = datetime.combine(today, out_time)

    early_out_minutes = (shift_end_dt - out_time_dt).total_seconds() / 60
    if early_out_minutes <= 0: return 0, 'On Time/Overtime', 0

    one_hour_leave_start_dt = datetime.combine(today, ONE_HOUR_LEAVE_START_TIME)
    one_hour_leave_end_dt = datetime.combine(today, ONE_HOUR_LEAVE_END_TIME)

    # 1 Hour Leave check (OUT time between 16:46 and 17:44)
    if out_time_dt >= one_hour_leave_start_dt and out_time_dt <= one_hour_leave_end_dt:
        return 1, '1 Hour Leave', 0

    evening_half_day_dt = datetime.combine(today, EVENING_HALF_DAY_TIME)
    # Evening Half Day Leave check (OUT time <= 16:45)
    if out_time_dt <= evening_half_day_dt:
        return 0, 'Half Day Leave (Evening OUT)', 1

    return 0, 'Other Early Out', 0


def get_day_classification(row):
    """Applies strict priority to generate a single, accurate daily status."""
    is_m_half = row['is_morning_half_day']
    is_e_half = row['is_evening_half_day']
    is_1hr_leave = row['early_out_units'] == 1
    late_units = row['late_units']

    # Priority 1: Full Day Leave 
    if is_m_half and is_e_half:
        return 'FULL Day Leave (Morn + Even Half)'
    # Priority 2: Half Day Leave 
    if is_m_half: return 'Half Day Leave (Morning IN)'
    if is_e_half: return 'Half Day Leave (Evening OUT)'
    # Priority 3: 1 Hour Leave
    if is_1hr_leave: return '1 Hour Leave (Early OUT)'
    # Priority 4: Late Units
    if late_units > 0: return f'{late_units} x 15 Min Late'
    # Priority 5: On Time / Incomplete Record
    if row['Time In'] is None or row['Time Out'] is None:
        return 'Incomplete Record'
    return 'On Time'


# =================================================================
# 3. FIXED DATA PARSING FUNCTION (Ensures correct date/time alignment)
# ** CACHED for Performance and Stability on File Re-upload **
# =================================================================

@st.cache_data(show_spinner="Processing uploaded report data...")
def restructure_attendance_data(df_raw):
    try:
        df_raw = df_raw.fillna('')
        df_raw[0] = df_raw[0].astype(str).str.strip()

        emp_rows = df_raw[df_raw[0].str.contains('^Employee:', regex=True, na=False)].index.tolist()
        days_row_index = df_raw[df_raw[0].str.contains('^Days$', regex=True, na=False)].index.max()

        if days_row_index is None: return pd.DataFrame()

        TIME_START_COL_INDEX = 2

        day_headers_raw = df_raw.iloc[days_row_index, TIME_START_COL_INDEX:].values
        day_headers_list_full = [str(h).strip() for h in day_headers_raw]

        final_data = []

        for emp_row_index in emp_rows:
            employee_str = df_raw.iloc[emp_row_index, 3]
            if str(employee_str).strip() == '': continue

            status_row_index = emp_row_index + 1
            intime_row_index = emp_row_index + 2
            outtime_row_index = emp_row_index + 3

            try:
                status_data = [str(c).strip() for c in df_raw.iloc[status_row_index, TIME_START_COL_INDEX:].values.tolist()]
                intime_data = [str(c).strip() for c in df_raw.iloc[intime_row_index, TIME_START_COL_INDEX:].values.tolist()]
                outtime_data = [str(c).strip() for c in df_raw.iloc[outtime_row_index, TIME_START_COL_INDEX:].values.tolist()]
            except IndexError:
                continue

            min_len = min(len(day_headers_list_full), len(status_data), len(intime_data), len(outtime_data))

            temp_df = pd.DataFrame({
                'Employee_Raw': employee_str,
                'Date/Day': day_headers_list_full[:min_len],
                'Status': status_data[:min_len],
                'InTime_Raw': intime_data[:min_len],
                'OutTime_Raw': outtime_data[:min_len]
            })

            temp_df['Employee Name'] = temp_df['Employee_Raw'].apply(lambda x: str(x).split(':')[-1].strip())
            temp_df['Time In'] = temp_df['InTime_Raw'].apply(to_time_safe)
            temp_df['Time Out'] = temp_df['OutTime_Raw'].apply(to_time_safe)

            if not temp_df.empty:
                final_data.append(temp_df)

        if not final_data: return pd.DataFrame()

        df_structured = pd.concat(final_data, ignore_index=True)

        df_clean = df_structured[(df_structured['Status'] == 'P')].copy()
        df_clean = df_clean[df_clean['Time In'].notna() | df_clean['Time Out'].notna()].copy()

        if df_clean.empty: return pd.DataFrame()

        return df_clean[['Employee Name', 'Date/Day', 'Time In', 'Time Out']]

    except Exception as e:
        st.error(f"A severe structural error occurred during file parsing: {e}")
        return pd.DataFrame()


def generate_employee_report(df_processed, employee_name):
    """Generates the final summary and detailed report for a specific employee."""
    employee_data = df_processed[df_processed['Employee Name'] == employee_name].copy()

    if employee_data.empty: return "No data found for this employee.", None

    employee_data[['late_units', 'late_classification', 'is_morning_half_day']] = employee_data['Time In'].apply(
        lambda x: pd.Series(calculate_late_status(x) if pd.notna(x) else (0, 'Missing IN Time', 0))
    )
    employee_data[['early_out_units', 'early_out_classification', 'is_evening_half_day']] = employee_data['Time Out'].apply(
        lambda x: pd.Series(calculate_early_out_status(x) if pd.notna(x) else (0, 'Missing OUT Time', 0))
    )

    employee_data['Day Classification'] = employee_data.apply(get_day_classification, axis=1)

    total_late_units = employee_data['late_units'].sum()
    total_1_hr_leaves = (employee_data['early_out_units'] == 1).sum()
    total_half_day_incidents = employee_data['is_morning_half_day'].sum() + employee_data['is_evening_half_day'].sum()

    summary = {
        'Total 15 Min Late Units': total_late_units,
        'Total Half Day Incidents': total_half_day_incidents,
        'Total 1 Hour Leaves': total_1_hr_leaves,
    }

    summary_df = pd.DataFrame(list(summary.items()), columns=['Category', 'Count'])

    report_details = employee_data[['Employee Name', 'Date/Day', 'Time In', 'Time Out', 'Day Classification']]

    return summary_df, report_details


# =================================================================
# 4. STREAMLIT APP LAYOUT & PREMIUM STYLING
# =================================================================

def app():
    st.set_page_config(layout="wide", page_title="Custom Attendance Dashboard")

    # --- Inject Custom CSS for Premium Look, Lavender Shade, and Bold Text ---
    custom_css = """
    <style>
    /* Premium Light Background (Beige/Cream) */
    .stApp, [data-testid="stAppViewBlockContainer"] {
        background-color: #FAF0E6 !important; /* Floral White - Light, premium feel */
    }
    
    /* Table Header Styling (Lavender and Bold) */
    .dataframe th {
        background-color: #E6E6FA !important; /* Lavender */
        font-weight: bold !important;
        color: #483D8B !important; /* Dark Slate Blue for contrast/bold look */
        padding: 10px 14px !important;
    }
    
    /* Table Data Cell Styling (Light Lavender Shade and Bold content) */
    .dataframe td {
        background-color: #F3E5F5 !important; /* Pale Lavender - consistent cell background */
        font-weight: bold !important; /* Make content bold */
        color: #000000 !important; /* Ensure text is visible */
        padding: 10px 14px !important;
        font-size: 14px !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    # -----------------------------------------------------------

    st.title("✅ Employee Work Duration & Late Report")
    
    # Sidebar Instructions
    st.sidebar.markdown("---")
    st.sidebar.subheader("Instructions")
    st.sidebar.markdown("1. Upload your report file.")
    st.sidebar.markdown("**2. If you modify the file, re-upload it to see changes.**")
    st.sidebar.markdown("---")
    
    # File Uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload WorkDurationReport (Excel/CSV)", 
        type=['xlsx', 'csv']
    )
    
    if uploaded_file is not None:
        try:
            # Check file extension and read accordingly
            if uploaded_file.name.endswith('.xlsx'):
                df_raw = pd.read_excel(uploaded_file, header=None)
            else: 
                # Assuming CSV, using latin1 for broader compatibility with different encodings
                df_raw = pd.read_csv(uploaded_file, header=None, encoding='latin1', skipinitialspace=True)

            st.sidebar.success("File uploaded and read successfully!")
            
            # Restructure the raw report data (Cached function runs here)
            df_processed = restructure_attendance_data(df_raw.copy())
            
            if df_processed.empty: 
                st.error("Could not parse employee data. Please ensure the file structure is correct.")
                return 

            # --- MODIFIED: Loop through all employees instead of using a dropdown ---
            employee_list = sorted(df_processed['Employee Name'].unique().tolist())
            
            for employee in employee_list:
                # Generate Report
                report_summary, report_details = generate_employee_report(df_processed, employee)
                
                # Report Section Header
                st.header(f"--- 👤 Report for **{employee}** ---")
                
                if report_summary is not None and report_details is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Lateness Summary") 
                        st.dataframe(report_summary, use_container_width=True)
                    
                    with col2:
                        st.subheader("Daily Attendance Classification")
                        st.dataframe(report_details, use_container_width=True)
                else:
                    st.warning(f"No valid attendance data found for {employee}.")

                st.markdown("---") # Separator after each employee report
            
            # --- END MODIFIED SECTION ---

        except Exception as e:
            st.error(f"An unexpected error occurred during processing. Error: {e}")

if __name__ == "__main__":
    # The application entry point
    app()
