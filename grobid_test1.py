import os
import requests
import subprocess
import time
import csv
from bs4 import BeautifulSoup
import logging
import atexit

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("grobid_extraction.log"),
                              logging.StreamHandler()])
logger = logging.getLogger()

# Global variable to store the GROBID process ID
grobid_container_id = None


# Function to check if GROBID server is running
def is_grobid_running():
    try:
        response = requests.get("http://localhost:8070/api/isalive", timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# Function to start GROBID server
def start_grobid_server():
    global grobid_container_id

    logger.info("GROBID server not running. Starting the server...")
    try:
        # Run Docker in detached mode
        result = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p", "8070:8070", "lfoppiano/grobid:0.8.1"],
            capture_output=True, text=True, check=True
        )

        # Store the container ID to stop it later
        grobid_container_id = result.stdout.strip()
        logger.info(f"GROBID server starting in background. Container ID: {grobid_container_id}")

        # Register the cleanup function to ensure container is stopped on exit
        atexit.register(stop_grobid_server)

        # Wait for server to initialize
        logger.info("Waiting for GROBID server to start...")
        attempts = 0
        while attempts < 12:  # Try for 2 minutes (12 * 10 seconds)
            if is_grobid_running():
                logger.info("GROBID server is now running!")
                return True
            logger.info(f"Waiting for GROBID server to start... Attempt {attempts + 1}/12")
            time.sleep(10)
            attempts += 1

        if not is_grobid_running():
            logger.error("Failed to start GROBID server after multiple attempts")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting GROBID server: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error starting GROBID server: {e}")
        return False


# Function to stop GROBID server
def stop_grobid_server():
    global grobid_container_id
    if grobid_container_id:
        try:
            logger.info(f"Stopping GROBID server container: {grobid_container_id}")
            subprocess.run(["docker", "stop", grobid_container_id], check=True)
            grobid_container_id = None
        except subprocess.CalledProcessError as e:
            logger.error(f"Error stopping GROBID container: {e}")
        except Exception as e:
            logger.error(f"Unexpected error stopping GROBID container: {e}")


# Function to extract abstract from PDF using GROBID with retries
def extract_abstract(pdf_path, max_retries=3):
    url = "http://localhost:8070/api/processFulltextDocument"

    # Set reasonable timeouts
    timeout = (30, 300)  # (connect timeout, read timeout) in seconds

    for attempt in range(max_retries):
        try:
            with open(pdf_path, 'rb') as pdf_file:
                files = {'input': pdf_file}
                params = {'consolidateHeader': '1'}

                # Send the PDF to GROBID for processing
                logger.info(
                    f"Sending request to GROBID for {os.path.basename(pdf_path)} (attempt {attempt + 1}/{max_retries})")
                response = requests.post(url, files=files, data=params, timeout=timeout)

                if response.status_code == 200:
                    xml_content = response.text
                    # Parse the XML response using BeautifulSoup
                    soup = BeautifulSoup(xml_content, 'xml')
                    # Extract the abstract from the XML content
                    abstract_tag = soup.find('abstract')

                    if abstract_tag:
                        return abstract_tag.text.strip()
                    else:
                        logger.warning(f"Abstract not found in the XML response for {os.path.basename(pdf_path)}")
                        return "Abstract not found."
                else:
                    logger.error(f"Error response from GROBID: {response.status_code} - {response.text[:100]}...")

            # If we get here, the request failed but didn't raise an exception
            logger.warning(f"Request failed with status code {response.status_code}. Retrying...")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")

            # Check if GROBID is still running
            if not is_grobid_running():
                logger.warning("GROBID server appears to have stopped. Attempting to restart...")
                stop_grobid_server()  # Make sure it's stopped
                if start_grobid_server():
                    logger.info("GROBID server restarted successfully.")
                else:
                    logger.error("Failed to restart GROBID server.")
                    return f"Error: GROBID server failure - {str(e)}"

            # If it's the last attempt, return the error
            if attempt == max_retries - 1:
                return f"Error: {str(e)}"

            # Otherwise wait and retry
            logger.info(f"Waiting before retry {attempt + 2}/{max_retries}...")
            time.sleep(10)  # Wait longer between retries

    # If we get here, all retries failed
    return "Error: Failed after multiple attempts"


# Function to process all PDFs in a folder and save results to CSV
def process_pdfs_in_folder(folder_path, output_csv='abstracts.csv'):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['PDF File', 'Abstract']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Count PDFs to process
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        total_pdfs = len(pdf_files)
        logger.info(f"Found {total_pdfs} PDF files to process")

        # Track progress
        processed = 0

        # Iterate over all files in the folder
        for filename in pdf_files:
            pdf_path = os.path.join(folder_path, filename)
            logger.info(f"Processing ({processed + 1}/{total_pdfs}): {filename}")

            try:
                # Extract abstract for the current PDF
                abstract = extract_abstract(pdf_path)
                # Write the result to CSV
                writer.writerow({'PDF File': filename, 'Abstract': abstract})
                processed += 1

                if "Error:" in abstract:
                    logger.warning(f"Extraction completed with error for {filename}: {abstract}")
                else:
                    logger.info(f"Abstract successfully extracted for {filename}")

            except Exception as e:
                logger.error(f"Unexpected error processing {filename}: {e}")
                writer.writerow({'PDF File': filename, 'Abstract': f"Error: Unexpected error - {str(e)}"})

            # Delay between requests to avoid overloading GROBID
            time.sleep(5)  # Increase delay to 5 seconds

            # Periodically check if GROBID is still running
            if processed % 10 == 0:
                if not is_grobid_running():
                    logger.warning("GROBID server appears to have stopped. Attempting to restart...")
                    stop_grobid_server()
                    if start_grobid_server():
                        logger.info("GROBID server restarted successfully.")
                    else:
                        logger.error("Failed to restart GROBID server. Stopping processing.")
                        break

    logger.info(f"All abstracts saved to {output_csv}. Processed {processed}/{total_pdfs} PDFs.")


# Main execution block
if __name__ == "__main__":
    folder_path = 'downloaded_pdfs'  # Replace with the path to your PDF folder

    # Ensure GROBID server is running
    if not is_grobid_running():
        if not start_grobid_server():
            logger.error("Failed to start GROBID server. Exiting.")
            exit(1)

    try:
        process_pdfs_in_folder(folder_path)
    finally:
        # Ensure we stop the GROBID server when done
        stop_grobid_server()