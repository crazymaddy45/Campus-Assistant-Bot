import re
import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter


data_folder = "Dataset"
output_file = "klnce_chunks.json"


ACRONYM_MAP = {
    # --- Core Departments ---1
    "AI & DS": "Artificial Intelligence and Data Science",
    "AIDS": "Artificial Intelligence and Data Science",
    "AI-DS": "Artificial Intelligence and Data Science",
    "CSE": "Computer Science and Engineering",
    "CS": "Computer Science",
    "IT": "Information Technology",
    "ECE": "Electronics and Communication Engineering",
    "EEE": "Electrical and Electronics Engineering",
    "EIE": "Electronics and Instrumentation Engineering",
    "MECH": "Mechanical Engineering",
    "CIVIL": "Civil Engineering",
    "BIO": "Biomedical Engineering",
    "BME": "Biomedical Engineering",
    "CHEM": "Chemical Engineering",
    "CHE": "Chemical Engineering",
    "AUTO": "Automobile Engineering",
    "MBA": "Master of Business Administration",
    "MCA": "Master of Computer Applications",
    "IOT": "Internet of Things",
    "CYBER SECURITY": "Cyber Security",
    "CSBS": "Computer Science and Business Systems",
    "AIML": "Artificial Intelligence and Machine Learning",
    "DS": "Data Science",
    "ML": "Machine Learning",
    "AI": "Artificial Intelligence",

    # --- Academic and Administrative Acronyms ---
    "HOD": "Head of the Department",
    "COE": "Controller of Examinations",
    "IQAC": "Internal Quality Assurance Cell",
    "NBA": "National Board of Accreditation",
    "NAAC": "National Assessment and Accreditation Council",
    "AICTE": "All India Council for Technical Education",
    "UGC": "University Grants Commission",
    "TNEA": "Tamil Nadu Engineering Admissions",
    "AU": "Anna University",
    "R&D": "Research and Development",
    "PG": "Postgraduate",
    "UG": "Undergraduate",
    "B.E": "Bachelor of Engineering",
    "B.TECH": "Bachelor of Technology",
    "M.E": "Master of Engineering",
    "M.TECH": "Master of Technology",
    "PhD": "Doctor of Philosophy",
    "RAC": "Research Advisory Committee",

    # --- Campus and Institutional ---
    "CDC": "Career Development Cell",
    "PLACEMENT CELL": "Placement and Training Cell",
    "PTA": "Parent Teachers Association",
    "NSS": "National Service Scheme",
    "NCC": "National Cadet Corps",
    "YRC": "Youth Red Cross",
    "WDC": "Women Development Cell",
    "EDC": "Entrepreneurship Development Cell",
    "IIC": "Institution Innovation Council",
    "ALUMNI": "Alumni Association",
    "CSI": "Computer Society of India",
    "ISTE": "Indian Society for Technical Education",
    "IEEE": "Institute of Electrical and Electronics Engineers",
    "IE": "Institution of Engineers",
    "IETE": "Institution of Electronics and Telecommunication Engineers",
    "SAE": "Society of Automotive Engineers",
    "ASME": "American Society of Mechanical Engineers",
    "ROBOTICS CLUB": "Robotics and Automation Club",
    "AI CLUB": "Artificial Intelligence Club",
    "CSE CLUB": "Computer Science Club",
    "INNOVATION CELL": "Innovation and Startup Cell",
}



def normalize_text(text):
    # Replace acronyms and clean unwanted characters
    for key, full_form in ACRONYM_MAP.items():
        text = re.sub(rf"\b{re.escape(key)}\b", full_form, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)  # normalize spaces
    text = re.sub(r"[^a-zA-Z0-9.,:;!?()\-\n ]", "", text)  # remove junk symbols
    return text.strip()

def extract_keywords(text):
    # Very lightweight keyword extraction
    words = re.findall(r'\b[A-Za-z]{4,}\b', text)
    common = ["department", "engineering", "college", "faculty", "student", "lab", "course"]
    keywords = [w.lower() for w in words if w.lower() not in common]
    return list(set(keywords[:10]))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)


docs = []

for file in os.listdir(data_folder):
    if file.endswith(".txt"):
        dept_name = file.replace(".txt", "")
        with open(os.path.join(data_folder, file), "r", encoding="utf-8") as f:
            text = f.read()

        text = normalize_text(text)

        # Merge small related headers like Vision, Mission, About into one chunk
        sections = re.split(r"(?<=:)\s*(?=[A-Z][a-z])", text)
        merged_text = " ".join(sections)

        chunks = text_splitter.split_text(merged_text)
        for i, chunk in enumerate(chunks):
            keywords = extract_keywords(chunk)
            docs.append({
                "id": f"{dept_name}_{i}",
                "department": dept_name,
                "text": chunk,
                "keywords": keywords,
                "aliases": [k for k, v in ACRONYM_MAP.items() if v.lower() in chunk.lower() or k.lower() in chunk.lower()]
            })

# Save enhanced chunks
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=2, ensure_ascii=False)

print(f"✅ Created {len(docs)} enhanced chunks from {len(os.listdir(data_folder))} department files.")