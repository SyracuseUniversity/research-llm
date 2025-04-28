import pandas as pd
import os
from database_handler import setup_research_info_table, insert_research_info

def combine_csvs():
    """
    Combines all provided CSV files into one DataFrame.
    Each CSV is tagged with a 'source' column.
    Then, only rows that correspond to downloaded PDFs are kept.
    A downloaded PDF is assumed to have a non-empty value in either the 'pdf_url'
    or 'final_work_url' column.
    """
    # List of CSV file paths to combine.
    csv_paths = [
        r"C:\Users\arapte\Downloads\Application\author_works.csv",
        r"C:\Users\arapte\Downloads\Application\open_alex_works.csv",
        r"C:\Users\arapte\Downloads\Application\openalex_final_works.csv",
        r"C:\Users\arapte\Downloads\Application\syracuse_university_orcid_data.csv",
        r"C:\Users\arapte\Downloads\cleaned_author_works.csv"
    ]
    
    dataframes = []
    for path in csv_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Tag each DataFrame with its source filename.
                df["source"] = os.path.basename(path)
                
                # Standardize columns to have at least 'researcher_name', 'work_title', and 'authors'
                if "author_name" in df.columns and "title" in df.columns:
                    df = df.rename(columns={"author_name": "researcher_name", "title": "work_title"})
                    if "authors" not in df.columns:
                        df["authors"] = df["researcher_name"]
                elif "display_name" in df.columns and "authors" in df.columns and "title" in df.columns:
                    df = df.rename(columns={"title": "work_title"})
                    # For these, extract the first name from the 'authors' field as the researcher name.
                    df["researcher_name"] = df["authors"].apply(lambda x: x.split(",")[0].strip() if isinstance(x, str) else "")
                elif "full_name" in df.columns and "work_title" in df.columns:
                    df = df.rename(columns={"full_name": "researcher_name"})
                    if "authors" not in df.columns:
                        df["authors"] = df["researcher_name"]
                dataframes.append(df)
            except Exception as e:
                print(f"Error reading {path}: {e}")
        else:
            print(f"File not found: {path}")
    
    if not dataframes:
        print("No CSV files were loaded.")
        return pd.DataFrame()

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Drop duplicate rows based on a unique work identifier if available.
    if "work_id" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=["work_id"])
    
    def is_downloaded(row):
        pdf = row.get("pdf_url", "")
        final_pdf = row.get("final_work_url", "")
        if isinstance(pdf, str):
            pdf = pdf.strip()
        if isinstance(final_pdf, str):
            final_pdf = final_pdf.strip()
        return (pdf != "") or (final_pdf != "")

    filtered_df = combined_df[combined_df.apply(is_downloaded, axis=1)].copy()
    return filtered_df

def populate_research_info_from_csv():
    """
    Populates the research_info table using data from the combined CSV.
    This function uses the standardized columns 'researcher_name', 'work_title', and 'authors'
    from only those CSV records that correspond to downloaded PDFs.
    It also constructs an 'info' string from additional fields if available.
    """
    setup_research_info_table()
    combined_df = combine_csvs()
    
    if combined_df.empty:
        print("No data available from CSV files after filtering for downloaded PDFs.")
        return

    total_records = len(combined_df)
    print(f"Populating research_info table with {total_records} records from combined CSVs (downloaded PDFs only)...")
    
    for idx, row in combined_df.iterrows():
        researcher_name = row.get("researcher_name", "")
        work_title = row.get("work_title", "")
        authors = row.get("authors", "")
        
        doi = ""
        if "doi_url" in row:
            doi = row.get("doi_url", "")
        elif "DOI_URL" in row:
            doi = row.get("DOI_URL", "")
        publication_date = row.get("publication_date", "")
        landing_url = row.get("landing_url", "")
        info_parts = []
        if pd.notna(doi) and str(doi).strip() != "":
            info_parts.append(f"DOI: {doi}")
        if pd.notna(publication_date) and str(publication_date).strip() != "":
            info_parts.append(f"Publication Date: {publication_date}")
        if pd.notna(landing_url) and str(landing_url).strip() != "":
            info_parts.append(f"Landing URL: {landing_url}")
        info = " | ".join(info_parts)
        
        insert_research_info(researcher_name, work_title, authors, info)
        print(f"[{idx+1}/{total_records}] Inserted research info for work: {work_title}")
    
    print("CSV-based research info population complete.")
