# import os
# import pandas as pd
# import requests
# import webbrowser
# from pathlib import Path
# import time

# # Define file paths
# csv_path = r'C:\Users\arapte\Downloads\author_works.csv'
# download_folder = Path(r'C:\codes\t5-db\fine_tuned_t5')

# # Ensure the download directory exists
# download_folder.mkdir(parents=True, exist_ok=True)

# # Load the CSV file
# try:
#     df = pd.read_csv(csv_path, sep='\t', dtype=str)  # Assuming tab-separated values
# except Exception as e:
#     print(f"Error reading CSV file: {e}")
#     exit()

# # Function to open the PDF URL
# def open_pdf_url(pdf_url):
#     try:
#         print(f"Opening: {pdf_url}")
#         webbrowser.open(pdf_url)
#         time.sleep(2)  # Pause to allow the browser to process
#     except Exception as e:
#         print(f"Failed to open {pdf_url}: {e}")

# # Iterate through the DataFrame and open PDF URLs
# for index, row in df.iterrows():
#     pdf_url = str(row.get('pdf_url', '')).strip()
    
#     if not pdf_url or not pdf_url.startswith("http"):
#         print(f"Skipping invalid or missing PDF URL: {pdf_url}")
#         continue
    
#     open_pdf_url(pdf_url)


# import pandas as pd

# # Define the file path
# file_path = r"C:\Users\arapte\Downloads\author_works.csv"  # Use raw string (r"") to avoid escape issues

# # Load the CSV file
# df = pd.read_csv(file_path)

# # Keep only the 'pdf_url' column
# df = df[['pdf_url']]

# # Save the modified CSV
# output_path = r"C:\Users\arapte\Downloads\filtered_author_works.csv"
# df.to_csv(output_path, index=False)

# print(f"Filtered CSV saved as '{output_path}'")



# import pandas as pd

# # Define the file path
# file_path = r"C:\Users\arapte\Downloads\filtered_author_works.csv"

# # Load the CSV file
# df = pd.read_csv(file_path)

# # Remove empty rows where 'pdf_url' is NaN or empty
# df = df.dropna(subset=['pdf_url'])  # Removes NaN values
# df = df[df['pdf_url'].str.strip() != '']  # Removes empty strings

# # Save the cleaned CSV
# output_path = r"C:\Users\arapte\Downloads\cleaned_author_works.csv"
# df.to_csv(output_path, index=False)

# print(f"Cleaned CSV saved as '{output_path}'")


import os
import pandas as pd
import requests

# Define paths
csv_path = r"C:\Users\arapte\Downloads\cleaned_author_works.csv"
download_dir = r"C:\codes\t5-db\fine_tuned_t5"

# Ensure the download directory exists
os.makedirs(download_dir, exist_ok=True)

# Load the CSV file
df = pd.read_csv(csv_path)

# Download each file
for index, row in df.iterrows():
    url = row['pdf_url']
    if pd.notna(url) and url.strip():  # Ensure the URL is valid
        file_name = os.path.join(download_dir, f"file_{index}.pdf")

        try:
            response = requests.get(url, stream=True, timeout=5)  # Set timeout to 5 seconds
            response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)

            with open(file_name, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"Downloaded: {file_name}")

        except requests.exceptions.Timeout:
            print(f"⚠️ Skipping (timeout): {url}")
        except requests.exceptions.HTTPError as e:
            if response.status_code in [403, 401]:  # Restricted access
                print(f"⚠️ Skipping (restricted access, manual download needed): {url}")
            else:
                print(f"⚠️ Failed to download {url}: HTTP Error {e}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Failed to download {url}: {e}")

print("Download process completed.")



# import os
# import time
# import pandas as pd
# import requests

# # Define paths
# csv_path = r"C:\Users\arapte\Downloads\cleaned_author_works.csv"
# download_dir = r"C:\codes\t5-db\fine_tuned_t5"

# # Ensure the download directory exists
# os.makedirs(download_dir, exist_ok=True)

# # Load the CSV file
# df = pd.read_csv(csv_path)

# # Custom headers to avoid blocking
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#     "Referer": "https://www.google.com/"  # Helps bypass some access blocks
# }

# # Create a session to persist headers and cookies
# session = requests.Session()
# session.headers.update(headers)

# # Log file for failed URLs
# failed_urls_path = os.path.join(download_dir, "failed_urls.txt")

# # Download each file
# for index, row in df.iterrows():
#     url = row['pdf_url']
#     if pd.notna(url) and url.strip():  # Ensure the URL is valid
#         file_name = os.path.join(download_dir, f"file_{index}.pdf")

#         # Skip if file already exists
#         if os.path.exists(file_name):
#             print(f"Skipping (already downloaded): {file_name}")
#             continue

#         # Detect URLs that require manual download
#         restricted_domains = [
#             "journals.aps.org", "link.aps.org",
#             "pnas.org", "sciencedirect.com", "wiley.com",
#             "royalsocietypublishing.org", "tandfonline.com",
#             "elementascience.org", "academic.oup.com"
#         ]

#         if any(domain in url for domain in restricted_domains):
#             print(f"⚠️ Skipping (restricted access, manual download needed): {url}")
#             with open(failed_urls_path, "a") as f:
#                 f.write(f"{url}\n")  # Save failed URLs for manual review
#             continue

#         # Retry with exponential backoff for 429 errors
#         attempt = 0
#         max_attempts = 5
#         delay = 5  # Start with 5 seconds delay

#         while attempt < max_attempts:
#             try:
#                 response = session.get(url, stream=True, allow_redirects=True)

#                 # Handle 403 Forbidden (Skip restricted journal PDFs)
#                 if response.status_code == 403:
#                     print(f"Skipping (403 Forbidden): {url}")
#                     with open(failed_urls_path, "a") as f:
#                         f.write(f"{url}\n")  # Log failed URLs
#                     break  # No need to retry

#                 # Handle 409 Conflict (Elementa Science, may require authentication)
#                 elif response.status_code == 409:
#                     print(f"⚠️ 409 Conflict Error (likely needs authentication): {url}")
#                     with open(failed_urls_path, "a") as f:
#                         f.write(f"{url}\n")  # Log failed URLs
#                     break  # Skip this file

#                 # Handle 429 Too Many Requests
#                 elif response.status_code == 429:
#                     print(f"⚠️ Too many requests (429). Retrying in {delay} seconds...")
#                     time.sleep(delay)
#                     delay *= 2  # Exponential backoff
#                     attempt += 1
#                     continue  # Retry

#                 response.raise_for_status()  # Raise an error for bad responses (other 4xx and 5xx)

#                 # Save the file
#                 with open(file_name, 'wb') as file:
#                     for chunk in response.iter_content(chunk_size=8192):
#                         file.write(chunk)

#                 print(f"✅ Downloaded: {file_name}")
#                 break  # Success, exit retry loop

#             except requests.exceptions.RequestException as e:
#                 print(f"⚠️ Failed to download {url}: {e}")
#                 with open(failed_urls_path, "a") as f:
#                     f.write(f"{url}\n")  # Log failed URLs
#                 break  # Stop retrying on permanent errors

# print("✅ Download process completed.")
# print(f"⚠️ Review {failed_urls_path} for URLs requiring manual download.")



