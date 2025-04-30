import requests
import pandas as pd
from xml.etree import ElementTree as ET

# Fetch ORCID data
def fetch_orcid_data(orcid_id):
    url = f"https://pub.orcid.org/v3.0/{orcid_id}"
    headers = {"Accept": "application/xml"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch data for ORCID ID {orcid_id}. Status code: {response.status_code}")
        return None

# Parse XML response
def parse_orcid_xml(orcid_id, xml_data):
    root = ET.fromstring(xml_data)

    # Define the namespaces for the XML
    namespaces = {
        'record': 'http://www.orcid.org/ns/record',
        'common': 'http://www.orcid.org/ns/common',
        'person': 'http://www.orcid.org/ns/person',
        'email': 'http://www.orcid.org/ns/email',
        'activities': 'http://www.orcid.org/ns/activities',
        'employment': 'http://www.orcid.org/ns/employment',
        'work': 'http://www.orcid.org/ns/work',
        'personal-details': 'http://www.orcid.org/ns/personal-details'
    }

    # Get full name (merge given-names and family-name)
    given_name = root.find('.//person:name/personal-details:given-names', namespaces)
    family_name = root.find('.//person:name/personal-details:family-name', namespaces)
    full_name = f"{given_name.text} {family_name.text}" if given_name is not None and family_name is not None else "N/A"

    # Get email (from <email:emails> and <email:email> tag)
    email_element = root.find('.//email:emails/email:email/email:email', namespaces)
    email_text = email_element.text if email_element is not None else "N/A"

    employment_data = []
    employments = root.findall('.//activities:employments//employment:employment-summary', namespaces)

    for employment in employments:
        org_name = employment.find('common:organization/common:name', namespaces)
        if org_name is not None and 'Syracuse University' in org_name.text:
            department = employment.find('common:department-name', namespaces)
            role = employment.find('common:role-title', namespaces)
            start_date = employment.find('common:start-date/common:year', namespaces)
            end_date = employment.find('common:end-date/common:year', namespaces)

            works = root.findall('.//activities:works//work:work-summary', namespaces)
            for work in works:
                work_title = work.find('work:title/common:title', namespaces)
                external_ids = work.findall('common:external-ids/common:external-id', namespaces)

                doi_url = None
                work_url = None
                arxiv_url = None

                for ext_id in external_ids:
                    id_type = ext_id.find('common:external-id-type', namespaces)
                    id_url = ext_id.find('common:external-id-url', namespaces)

                    if id_type is not None and id_url is not None:
                        if id_type.text.lower() == 'arxiv':
                            arxiv_url = id_url.text
                        elif id_type.text.lower() == 'doi':
                            doi_url = id_url.text
                        else:
                            work_url = id_url.text

                # Prefer arXiv if available, otherwise fallback to work_url
                preferred_work_url = arxiv_url if arxiv_url else (work_url if work_url else "N/A")

                employment_data.append({
                    'orcid_id': orcid_id,
                    'full_name': full_name,
                    'email': email_text,
                    'employment': org_name.text,
                    'department': department.text if department is not None else "N/A",
                    'role': role.text if role is not None else "N/A",
                    'start_year': start_date.text if start_date is not None else "N/A",
                    'end_year': end_date.text if end_date is not None else "N/A",
                    'work_title': work_title.text if work_title is not None else "N/A",
                    'DOI_URL': doi_url if doi_url else "N/A",
                    'work_url': preferred_work_url
                })

    return employment_data

# Remove duplicate DOI_URL rows
def remove_duplicate_doi_rows(df):
    # Sort by work_url so that rows with non-"N/A" work_url come first
    df = df.sort_values(by=['DOI_URL', 'work_url'], ascending=[True, False])
    # Drop duplicate DOI_URL, keeping the first occurrence
    return df.drop_duplicates(subset='DOI_URL', keep='first')

# Fetch and store data
def fetch_and_store_data(df):
    all_data = []
    for _, row in df.iterrows():
        orcid_id = row['orcid']
        xml_data = fetch_orcid_data(orcid_id)
        if xml_data:
            employment_data = parse_orcid_xml(orcid_id, xml_data)
            all_data.extend(employment_data)

    result_df = pd.DataFrame(all_data)

    # Remove duplicate DOI_URL rows
    result_df = remove_duplicate_doi_rows(result_df)

    # Save to CSV
    result_df.to_csv('syracuse_university_orcid_data.csv', index=False)
    print("Data saved to syracuse_university_orcid_data.csv")

# Read data from merged researchers data
df = pd.read_csv('orcid_data/merged_syracuse_data.csv')

fetch_and_store_data(df)
