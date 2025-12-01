import pandas as pd

# 1. Read the original Excel file
input_file = "Courses_data.xlsx"     # change path if needed
df = pd.read_excel(input_file)

# 2. Keep only the desired columns
cols_to_keep = ["COURSE_ID", "NAME_E", "DESCRIP_EFULL"]
df_filtered = df[cols_to_keep]

# 3. Rename columns to new names
new_column_names = {
    "COURSE_ID": "COURSE_ID",        # keep same or change if you want
    "NAME_E": "COURSE_NAME_EN",
    "DESCRIP_EFULL": "COURSE_DESC_EN"
}
df_filtered = df_filtered.rename(columns=new_column_names)

# 4. Save to a new Excel file
output_file = "Courses_data_cleaned.xlsx"
df_filtered.to_excel(output_file, index=False)

print("Done! Saved cleaned file as:", output_file)
