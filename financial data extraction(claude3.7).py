# Group imports by logical modules
import os
import numpy as np
import pandas as pd
from tabulate import tabulate
import fitz  # PyMuPDF
import re
import sys


from pdf2image import convert_from_path
import pytesseract
import camelot
import cv2
from PIL import Image
import tempfile

# Anthropic 客户端
import anthropic

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

def get_pdf_path():
    """Get PDF file path from user input"""
    pdf_path = input("Please enter the path to the PDF file: ")
    return pdf_path


def convert_pdf_to_images(pdf_path, dpi=300):
    """Convert PDF pages to images"""
    print("\nStep 1: Converting PDF to images...")
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Successfully converted {len(images)} pages to images")
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []


def perform_ocr(images):
    """Perform OCR on images"""
    print("\nStep 2: Performing OCR on images...")
    ocr_texts = []

    if not images:
        print("No images available for OCR. Skipping OCR step.")
        return ocr_texts

    for i, img in enumerate(images):
        try:
            text = pytesseract.image_to_string(img, lang='eng')
            ocr_texts.append(text)
            print(f"OCR completed for page {i + 1}")
        except Exception as e:
            print(f"Error in OCR processing for page {i + 1}: {e}")
            ocr_texts.append("")
    return ocr_texts


def extract_tables(pdf_path):
    """Extract tables from PDF using Camelot and return a list of DataFrames."""
    print("\nStep 3: Extracting tables from PDF...")
    dataframes = []
    try:
        extracted_tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        print(f"Found {len(extracted_tables)} tables in PDF")
        for i, table in enumerate(extracted_tables):
            df = table.df
            # 假设第一行为标题，将其设置为列名并删除第一行
            df.columns = df.iloc[0]
            df = df[1:]
            dataframes.append(df)
    except Exception as e:
        print(f"Error extracting tables: {e}")
    return dataframes


def extract_indicators_from_dataframe(df, target_year, extraction_params):
    """
    Extracts financial indicators from the DataFrame using keyword matching in the target column.
    If no matching value is found, returns "Value not found".

    Parameters:
        df: pandas DataFrame containing the table data.
        target_year: A string (e.g., "2024") representing the target column.
        extraction_params: A dictionary mapping each indicator to a list of potential keywords.

    Returns:
        A dictionary where keys are indicator names and values are the extracted value
        or "Value not found" if no match is found.
    """
    results = {}
    if target_year not in df.columns:
        for indicator in extraction_params:
            results[indicator] = "Value not found"
        return results

    for indicator, keywords in extraction_params.items():
        found_value = None
        for idx, row in df.iterrows():
            cell_text = str(row.iloc[0]).lower()
            for keyword in keywords:
                if keyword.lower() in cell_text:
                    found_value = row[target_year]
                    break
            if found_value is not None:
                break
        results[indicator] = found_value if found_value is not None else "Value not found"
    return results


def extract_text_from_pdf(pdf_path):
    """Extract text directly from PDF using PyMuPDF"""
    print("\nStep 4: Extracting text directly from PDF...")
    text_content = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            text_content.append(text)
            print(f"Extracted text from page {i + 1}")
        doc.close()
    except Exception as e:
        print(f"Error extracting text: {e}")
    return text_content


def merge_and_clean_content(ocr_texts, tables, text_content):
    """Merge and clean all extracted content"""
    print("\nStep 5: Merging and cleaning all extracted content...")
    combined_text = ""
    combined_text += "=== OCR EXTRACTED TEXT ===\n\n"
    for i, text in enumerate(ocr_texts):
        combined_text += f"[OCR PAGE {i + 1}]\n{text}\n\n"
    combined_text += "=== EXTRACTED TABLES ===\n\n"
    for table in tables:
        combined_text += f"{table}\n"
    combined_text += "=== DIRECT PDF TEXT ===\n\n"
    for i, text in enumerate(text_content):
        combined_text += f"[PDF PAGE {i + 1}]\n{text}\n\n"
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', combined_text)
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', cleaned_text)
    print(f"Combined and cleaned text - Total length: {len(cleaned_text)} characters")
    return cleaned_text


def format_for_llm(cleaned_text):
    """Format content for LLM analysis"""
    print("\nStep 6: Formatting data for LLM...")
    formatted_text = """
    === FINANCIAL REPORT CONTENT ===

    The following content has been extracted from a financial report using multiple methods:
    1. OCR text extracted from PDF pages
    2. Tables extracted from the PDF
    3. Direct text extraction from the PDF

    Please analyze this content to extract the requested financial information.

    """
    formatted_text += cleaned_text
    print(f"Formatted text prepared for LLM analysis - Length: {len(formatted_text)} characters")
    return formatted_text

#chunk包含的字符增加, 4月25日修改
def create_document_chunks(formatted_text, chunk_size=20000, overlap=1000):
    """Split the formatted text into document chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = text_splitter.split_text(formatted_text)
    print(f"Document split into {len(chunks)} chunks")
    return chunks


def print_document_info(doc_text):
    """Print document information statistics"""
    print("\n" + "=" * 50)
    print(f"{'Document Information Statistics':^46}")
    print("=" * 50)
    print(f"Total document length: {len(doc_text)} characters")
    word_count = len(doc_text.split())
    print(f"Approximate word count: {word_count:,}")
    print("=" * 50 + "\n")


def analyze_financial_data(api_key, text_chunks, financial_analysis_query):
    """
    Analyze financial data using Anthropic's Claude model.
    """
    print("\n" + "-" * 50)
    print(f"{'Starting Financial Data Analysis':^46}")
    print("-" * 50)

    # 初始化 Anthropic 客户端
    client = anthropic.Anthropic(api_key=api_key)

    # 系统提示
    system_prompt = """You are a senior financial analyst with expertise in banking sector analysis. 
    Your task is to extract financial data from bank reports and compute key financial ratios."""

    all_results = []

    # 针对每个文本块调用 Claude 接口
    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i + 1} of {len(text_chunks)}...")
        try:
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1024,
                temperature=0.0,
                system=system_prompt,  # 将系统提示作为顶层参数, 4月25日修改
                messages=[
                    {"role": "user", "content": f"{financial_analysis_query}\n\n{chunk}"}
                ],
            )

            result = response.content[0].text

            all_results.append(result)
            print(f"Analysis completed for chunk {i + 1}")
            print(f"Chunk {i + 1} raw output:\n{result}\n{'-' * 40}") #4月25日更改

        except Exception as e:
            print(f"Error analyzing chunk {i + 1}: {e}")
            all_results.append(f"Error in analysis: {str(e)}")

    combined_analysis = "\n\n===== CHUNK DIVIDER =====\n\n".join(all_results)

    # 如果有多个块，进行最终综合
    if len(text_chunks) > 1:
        try:
            print("Performing final synthesis of all results...")
            synthesis_prompt = """You are a senior financial analyst. You have analyzed multiple chunks of a financial report.
            Below are the separate analyses from each chunk. Your task is to synthesize these results into a single coherent analysis.
                
            For each required financial indicator:
            1. Identify the most reliable value from all chunks
            2. If there are conflicting values, choose the one with the clearest source citation
            3. For computed ratios, recalculate based on your selected indicator values
                
            Present your final analysis in the requested format with all extracted values and computed ratios."""

            # 使用新版 API 格式进行最终分析
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                system=system_prompt,  # ← 新增4月25日修改
                messages=[
                    {"role": "user", "content": f"{synthesis_prompt}\n\n{combined_analysis}"}
                ],
                max_tokens=1024,
                temperature=0.0
            )

            final_result = response.content[0].text
        except Exception as e:
            print(f"Error in final synthesis: {e}")
            final_result = combined_analysis
    else:
        final_result = combined_analysis

    return final_result


def format_financial_results(result):
    """Format financial analysis results as tables"""
    financial_data = str(result)

    # 添加调试输出，查看原始结果
    print("\nRaw result for formatting:")
    print(financial_data)

    extracted_values = []
    computed_ratios = []
    in_extracted_values = False
    in_computed_ratios = False

    for line in financial_data.split('\n'):
        title = line.strip()
        # 更灵活地匹配章节标题
        if re.search(r'(?i)#?\s*extracted\s+values[:：]?', title):
            in_extracted_values = True
            in_computed_ratios = False
            continue
        elif re.search(r'(?i)#?\s*computed\s+ratios[:：]?', title):
            in_extracted_values = False
            in_computed_ratios = True
            continue
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            sentence_info = ""
            if '(from' in value and ')' in value:
                main_value = value.split('(from')[0].strip()
                sentence_info = "(from" + value.split('(from')[1].strip()
                value = f"{main_value} {sentence_info}"
            if in_extracted_values:
                extracted_values.append([key, value])
            elif in_computed_ratios:
                computed_ratios.append([key, value])
    print("\n" + "=" * 80)
    print(f"{'Financial Data Analysis Results':^76}")
    print("=" * 80)
    if extracted_values:
        print("\n" + "-" * 80)
        print(f"{'Extracted Financial Indicators':^76}")
        print("-" * 80)
        print(tabulate(extracted_values, headers=["Indicator Name", "Value"], tablefmt="grid"))
    else:
        print("\nNo extracted values found in the results.")
    if computed_ratios:
        print("\n" + "-" * 80)
        print(f"{'Calculated Financial Ratios':^76}")
        print("-" * 80)
        print(tabulate(computed_ratios, headers=["Ratio Name", "Value"], tablefmt="grid"))
    else:
        print("\nNo computed ratios found in the results.")
    print("=" * 80 + "\n")


extraction_params = {
    "Net Income": ["Net Income", "Profit", "Net Profit", "Net Earnings", "Profit After Tax"],
    "Total Assets": ["Total Assets", "Assets", "Balance Sheet Total", "Total Assets on Balance Sheet"],
    "Shareholders' Equity": ["Shareholders' Equity", "Total Equity", "Net Assets", "Total Shareholders' Funds"]
}


# 主函数
def main():
    """Main function"""
    api_key = input("Please enter your Anthropic API key: ").strip()
    if not api_key:
        print("Warning: No API key provided. API calls will fail.")

    # Step 1: Get PDF path
    print("\n" + "*" * 50)
    print(f"{'Enhanced Financial Report Analysis Tool':^46}")
    print("*" * 50)
    pdf_path = get_pdf_path()
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' does not exist.")
        return

    # Step 2: Convert PDF to images
    images = convert_pdf_to_images(pdf_path)
    # Step 3: Perform OCR on images
    ocr_texts = perform_ocr(images)
    # Step 4: Extract tables from PDF
    tables = extract_tables(pdf_path)
    # Step 5: Extract text directly from PDF
    text_content = extract_text_from_pdf(pdf_path)

    # 新步骤：从结构化表格中提取财务指标
    extracted_dfs = extract_tables(pdf_path)
    target_year = "2024"
    overall_results = {}
    for df in extracted_dfs:
        results = extract_indicators_from_dataframe(df, target_year, extraction_params)
        for indicator, value in results.items():
            if indicator not in overall_results and value != "Value not found":
                overall_results[indicator] = value
    for indicator in extraction_params.keys():
        if indicator not in overall_results:
            overall_results[indicator] = "Value not found"
    print("Extracted financial indicators for", target_year, ":", overall_results)

    # Step 6: Merge and clean all content
    cleaned_text = merge_and_clean_content(ocr_texts, tables, text_content)
    # Step 7: Format for LLM
    formatted_text = format_for_llm(cleaned_text)
    print_document_info(formatted_text)
    # Step 8: Create document chunks
    text_chunks = create_document_chunks(formatted_text)

    # Step 9: Prepare financial analysis query
    financial_analysis_query = """
    You are a senior financial analyst with expertise in banking sector analysis. Your task is to carefully extract financial data from the provided bank's annual report and compute key financial ratios with precision. Use ONLY 2024 data for your extraction. Ignore the other years.
    
    The following table is a sample structure representing the format that appears in the document:
    
    ------------------------------
    | Indicator           | 2023         | 2024             |
    |---------------------|--------------|------------------|
    | Total Assets        | 1,805,299    | Value not found  |
    | Net Income          | 225,000      | 225,000          |
    | Shareholders' Equity| 700,000      | 700,000          |
    ------------------------------
    
    EXTRACTION GUIDELINES:
    1. Be thorough in searching for data across the entire document.
    2. Pay special attention to sections titled: "Financial Highlights", "Key Financial Indicators", "Balance Sheet", "Income Statement", "Financial Review", or "Capital Management".
    3. Recognize that financial indicators may appear under various synonyms:
    - Net Income: Also "Profit", "Net Profit", "Net Earnings", "Profit After Tax"
    - Total Assets: Also "Assets", "Balance Sheet Total", "Total Assets on Balance Sheet"
    - Shareholders' Equity: Also "Total Equity", "Net Assets", "Total Shareholders' Funds"
    4. ALWAYS convert all monetary values to millions (1,000,000) of the report's currency.
    5. For each value extracted, include the EXACT sentence where it appears in quotation marks.
    6. If multiple values are found, prioritize data explicitly labeled for 2024.
    7. If a ratio cannot be calculated due to missing data, output "Cannot compute, missing data".
    8. Output all monetary values in millions with "million" explicitly stated.
    9. Extract data only from the column with header "2024".
    10. Double-check all extracted values for accuracy before calculating ratios.
    
    OUTPUT FORMAT:
    
    Extracted Values:
    
    Net Income: <value in millions> million (from "<exact source sentence>")
    Total Assets: <value in millions> million (from "<exact source sentence>")
    Shareholders' Equity: <value in millions> million (from "<exact source sentence>")
    
    Computed Ratios:
    
    ROA: <percentage with 2 decimal places>
    ROE: <percentage with 2 decimal places>
    
    """
    # Step 10: Analyze data using Anthropic Claude
    result = analyze_financial_data(api_key, text_chunks, financial_analysis_query)
    # Step 11: Format results
    format_financial_results(result)
    print("\nFinancial report analysis completed successfully!")


if __name__ == "__main__":
    main()

