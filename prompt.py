SYSTEMPROMPT = '''You are an experienced critical care physician working in an Intensive Care Unit (ICU). You are skilled in interpreting ICU discharge summaries of patients, and predicting clinical outcomes based on the patient's status at the time of discharge and their overall hospital course.'''

INSTRUCTION_PROMPT = {
    'diagnosis_hierarchical': '''\
Your primary task is to assess the patient’s complete longitudinal ICU record—**all prior and current ICU discharge summaries**—to predict which diagnosis categories (from the predefined candidate set) are most likely to be coded at the patient’s next hospital visit.

The diagnostic taxonomy is hierarchical. Use the following structure:

**Parent diagnostic systems** (e.g., "Diseases of the Circulatory System")  
→ **Child-level clinical subcategories** (e.g., "Cardiac Conditions", "Cerebrovascular Disorders")

Use the following clinical features to guide your prediction:

- Longitudinal comorbidities (e.g., CHF, diabetes, COPD)  
- Acute-on-chronic exacerbations and organ failure  
- New or recurrent infections  
- ICU interventions (e.g., ventilation, dialysis, vasopressors)  
- Functional decline or unresolved complications at discharge  
- Trends in labs, imaging, or organ support  
- Discharge disposition (e.g., home, rehab, hospice)

---

### Candidate Parent Diagnostic Systems

- Perinatal and Congenital Conditions
- Diseases of the Blood and Immune System
- Diseases of the Circulatory System  
- Diseases of the Respiratory System  
- Diseases of the Digestive System  
- Diseases of the Genitourinary System  
- Pregnancy; Childbirth; and Postpartum Complications  
- Diseases of the Musculoskeletal and Connective Tissue  
- Diseases of the Nervous System and Sense Organs  
- Endocrine; Nutritional; and Metabolic Diseases 
- Infectious and Parasitic Diseases  
- Diseases of the Skin and Subcutaneous Tissue  
- Injury; Poisoning; and External Causes  
- Symptoms, Signs, and Other Conditions
- Mental; Behavioral; and Neurodevelopmental Disorder  
- Neoplasms  
- External Causes of Morbidity (E-Codes)

---

### Allowed Child Subcategories

You must select one or more child-level subcategories **only** from this predefined mapping:

{
  "Perinatal and Congenital Conditions": [
    "Neonatal Trauma and Injury",
    "Other Perinatal Conditions",
    "Congenital Anomalies"
  ],
  "Diseases of the Blood and Immune System": [
    "Anemia and Hematologic Disorders",
    "Immunologic Disorders"
  ],
  "Diseases of the Circulatory System": [
    "Heart Diseases",
    "Hypertensive Diseases",
    "Cerebrovascular Disorders",
    "Peripheral and Venous Diseases"
  ],
  "Diseases of the Respiratory System": [
    "Respiratory Infections",
    "Chronic and Obstructive Pulmonary Diseases",
    "Other Respiratory Conditions"
  ],
  "Diseases of the Digestive System": [
    "Upper Gastrointestinal Disorders",
    "Lower Gastrointestinal and Abdominal Disorders",
    "Hepatic and Pancreatic Disorders",
    "Oral and Dental Conditions"
  ],
  "Diseases of the Genitourinary System": [
    "Renal and Urinary Tract Disorders",
    "Reproductive Disorders"
  ],
  "Pregnancy; Childbirth; and Postpartum Complications": [
    "Labor and Delivery Complications",
    "Postpartum and Puerperal Complications"
  ],
  "Diseases of the Musculoskeletal and Connective Tissue": [
    "Autoimmune and Connective Tissue Disorders,
    "Skeletal and Acquired Musculoskeletal Disorders"
  ]
  "Diseases of the Nervous System and Sense Organs": [
    "Central Nervous System Disorders",
    "Sensory and Vestibular Disorders"
  ],
  "Endocrine; Nutritional; and Metabolic Diseases": [
    "Endocrine and Diabetic Disorders",
    "Nutritional and Metabolic Disorders"
  ],
  "Infectious and Parasitic Diseases": [
    "Bacterial and Septic Infections",
    "Viral; Mycotic; and Other Infections",
    "Sexually Transmitted and Preventive Conditions"
  ],
  "Diseases of the Skin and Subcutaneous Tissue": [
    "Inflammatory and Infectious Skin Disorders"
  ],
  "Injury; Poisoning; and External Causes": [
    "Physical Trauma and Injuries",
    "Toxicological and Iatrogenic Complications"
  ],
  "Symptoms, Signs, and Other Conditions": [
    "Symptoms and Signs",
    "Aftercare and Other Issues"
  ],
  "Mental; Behavioral; and Neurodevelopmental Disorders": [
    "Neurodevelopmental and Pediatric Disorders",
    "Mood; Anxiety; and Cognitive Disorders",
    "Psychotic and Substance Use Disorders",
    "Other Conditions and Events"
  ],
  "Neoplasms": [
    "Gastrointestinal Cancers",
    "Head; Neck; and Thoracic Cancers",
    "Urogenital and Reproductive Cancers",
    "Hematologic and Endocrine Cancers",
    "Cancers of Other Systems",
    "Benign Neoplasms",
    "Unspecified Neoplasms and Other Conditions"
  ],
  "External Causes of Morbidity (E-Codes)": [
    "Environmental; Mechanical; and Intentional Injuries"
  ]
}

For each selected **parent-level diagnostic system**, you **MUST** select **one or more** clinically plausible child-level diagnosis categories. If the patient’s record supports multiple complications, sequelae, or chronic comorbidities within a system, include **all applicable child categories**. Avoid under-selecting. Draw upon clinical history, trends, and discharge markers.

---

### Step-by-Step Reasoning Process

1. **Timeline Synthesis**: Describe how the patient’s health has evolved across ICU encounters (e.g., persistent CHF, recent sepsis, renal decline).  
2. **Diagnostic Trait Inference**: Infer clinical risks based on underlying disease traits or complications (e.g., steroid use → infection risk, multiple admissions → critical illness).  
3. **Parent-Level Filtering**: Select all high-level diagnostic systems relevant to the patient’s longitudinal profile.  
4. **Subcategory Disambiguation**: For each selected parent, choose **one or more** matching child-level categories, based strictly on the allowed schema.  
5. **Prediction**: Output a JSON dictionary mapping each parent to its most plausible child categories.

---

### Output Format

Return a single JSON object with **exactly two keys**:

1. `"think"`: A clinical reasoning narrative (less than 150 words), including timeline synthesis, pathophysiological reasoning, and justification for each selected parent and subcategory.  
2. `"answer"`: A dictionary mapping each predicted **parent-level diagnostic system** to a **list of one or more** predicted child-level categories.

---

### Example Output:
```json
{
  "think": "This patient has had multiple ICU admissions over 18 months for acute decompensated heart failure with reduced ejection fraction and atrial fibrillation, complicated by pulmonary edema and cardiogenic shock requiring inotropes. Renal function shows progressive chronic kidney disease with episodes of acute tubular injury during volume shifts. The course has been further complicated by recurrent MRSA pneumonia necessitating intubation and prolonged antibiotics, as well as chronic steroid use for autoimmune vasculitis. At the most recent discharge, the patient had unresolved fluid overload and borderline oxygenation and was transferred to a skilled nursing facility. Overall, the patient remains at high risk for recurrent cardiac decompensation, infectious complications, worsening renal failure, and continued critical illness.",
  "answer": {
    "Diseases of the Circulatory System": [
      "Heart Diseases",
      "Peripheral and Venous Diseases"
    ],
    "Diseases of the Respiratory System": [
        "Respiratory Infections"
    ],
    "Diseases of the Genitourinary System": [
      "Renal and Urinary Tract Disorders"
    ],
    "Infectious and Parasitic Diseases": [
      "Bacterial and Septic Infections",
      "Viral; Mycotic; and Other Infections"
    ]
  }
}
```
'''
}