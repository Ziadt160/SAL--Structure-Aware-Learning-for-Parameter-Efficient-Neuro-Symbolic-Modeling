from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        # Logo could go here
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Evoth Labs', 0, 1, 'R')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def create_pdf(md_file, pdf_file):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Read Markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        # Sanitize for latin-1
        line = line.replace('\u2019', "'").replace('\u2018', "'").replace('\u2013', "-").replace('\u2014', "--")
        if not line:
            pdf.ln(5)
            continue
            
        # Headings
        if line.startswith('# '):
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, line[2:], 0, 1, 'L')
            pdf.set_font("Arial", size=12)
        elif line.startswith('## '):
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, line[3:], 0, 1, 'L')
            pdf.set_font("Arial", size=12)
        elif line.startswith('### '):
            pdf.ln(2)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, line[4:], 0, 1, 'L')
            pdf.set_font("Arial", size=12)
            
        # Bold text (Key-Value style in MD often)
        elif '**' in line and line.count('**') >= 2:
            # Simple bold parser for "Key: Value" lines
            parts = line.split('**')
            # Assuming format: **Bold**: text or Text **Bold** Text
            # This is a naive parser for the generated report structure
            pdf.set_font("Arial", size=12) 
            # Check if it's a list item
            prefix = ""
            if line.startswith('* '): 
                prefix = "*" 
                pdf.set_x(15)
            elif line.startswith('- '):
                 prefix = "-"
                 pdf.set_x(15)
                 
            clean_line = line.replace('* ', '').replace('- ', '').replace('**', '')
            pdf.multi_cell(0, 6, prefix + " " + clean_line)
            
        # List items
        elif line.startswith('* ') or line.startswith('- '):
            pdf.set_x(15)
            pdf.multi_cell(0, 6, chr(149) + " " + line[2:])
            
        # Table row (Markdown tables)
        elif '|' in line:
            if '---' in line: continue
            col_width = pdf.w / 4.5
            cols = [c.strip() for c in line.split('|') if c.strip()]
            if len(cols) > 0:
                # Naive table
                pdf.set_font("Arial", 'B' if 'Metric' in line else '', 10)
                for col in cols:
                    pdf.cell(col_width, 6, col, 1)
                pdf.ln()
                pdf.set_font("Arial", size=12)
            
        # Normal text
        else:
            pdf.multi_cell(0, 6, line)
            
    pdf.output(pdf_file)
    print(f"PDF Generated: {pdf_file}")

if __name__ == "__main__":
    # Source path is known from previous steps
    base_dir = r"C:\Users\20102\.gemini\antigravity\brain\e208e69b-33d7-4cc9-b187-c5baa3cdd90b"
    md_path = os.path.join(base_dir, "use_case_architecture_screening.md")
    pdf_out = os.path.join(base_dir, "Evoth_Use_Case_Report.pdf")
    
    create_pdf(md_path, pdf_out)
