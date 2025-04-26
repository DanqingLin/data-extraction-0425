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
    Extracts environmental indicators from the DataFrame using keyword matching in the target column.
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
    === REPORT CONTENT ===

    The following content has been extracted from a report using multiple methods:
    1. OCR text extracted from PDF pages
    2. Tables extracted from the PDF
    3. Direct text extraction from the PDF

    Please analyze this content to extract the requested environmental information.

    """
    formatted_text += cleaned_text
    print(f"Formatted text prepared for LLM analysis - Length: {len(formatted_text)} characters")
    return formatted_text


# chunk包含的字符增加, 4月25日修改
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


def analyze_environmental_data(api_key, text_chunks, environmental_analysis_query):
    """
    Analyze environmental data using Anthropic's Claude model.
    """
    print("\n" + "-" * 50)
    print(f"{'Starting Environment Data Analysis':^46}")
    print("-" * 50)

    # 初始化 Anthropic 客户端
    client = anthropic.Anthropic(api_key=api_key)

    # 系统提示
    system_prompt = """You are a senior environmental analyst specializing in corporate sustainability reporting. 
    Your task is to extract environmental data from corporate annual reports and ensure all metrics are standardized with consistent units."""

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
                    {"role": "user", "content": f"{environmental_analysis_query}\n\n{chunk}"}
                ],
            )

            result = response.content[0].text

            all_results.append(result)
            print(f"Analysis completed for chunk {i + 1}")
            print(f"Chunk {i + 1} raw output:\n{result}\n{'-' * 40}")  # 4月25日更改

        except Exception as e:
            print(f"Error analyzing chunk {i + 1}: {e}")
            all_results.append(f"Error in analysis: {str(e)}")

    combined_analysis = "\n\n===== CHUNK DIVIDER =====\n\n".join(all_results)

    # 如果有多个块，进行最终综合
    if len(text_chunks) > 1:
        try:
            print("Performing final synthesis of all environmental results...")
            synthesis_prompt = """You are a senior environmental analyst specializing in corporate sustainability reporting. You have analyzed multiple chunks of an annual report.
            Below are the separate analyses from each chunk. Your task is to synthesize these results into a single coherent environmental analysis.

            For each required environmental metric:
            1. Identify the most reliable value from all chunks
            2. If there are conflicting values, choose the one with the clearest source citation
            3. Ensure all units are standardized according to the guidelines (tonnes CO2e for emissions, MWh for energy, etc.)

            Present your final analysis in the requested format with all extracted environmental metrics:

            Extracted Environmental Metrics:
            Total Global GHG Emissions (Scope 1, 2 & 3 - Location-Based): <value in tonnes CO2e> tonnes CO2e (from "<exact source sentence>")
            Net Global GHG Emissions (Scope 1, 2 & 3 - Market-Based): <value in tonnes CO2e> tonnes CO2e (from "<exact source sentence>")
            Total Absolute Financed Emissions: <value in MtCO₂-e> MtCO₂-e (from "<exact source sentence>")
            Total Premises Energy Use Consumed: <value in MWh> MWh (from "<exact source sentence>")
            Total Renewable Energy Consumption: <value in MWh> MWh (from "<exact source sentence>")
            Net Energy Consumption: <value in MWh> MWh (from "<exact source sentence>")
            Total Road Transport Energy Use: <value in MWh> MWh (from "<exact source sentence>")
            Total Paper Use: <value in tonnes> tonnes (from "<exact source sentence>")
            Total Water Consumption: <value in KL> KL (from "<exact source sentence>")
            Waste Recycling Rate: <percentage> % (from "<exact source sentence>")
            Total Sustainable Finance Deals: <value in millions> million (from "<exact source sentence>")
            Products for Personal and Business Customers Supporting Low-Carbon Transition: <value in $m> $m (from "<exact source sentence>")"""

            # 使用新版 API 格式进行最终分析
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                system=system_prompt,  # ← 新增 4月25日修改
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


def format_environmental_results(result):
    """Format environmental analysis results as tables"""
    environmental_data = str(result)

    # 添加调试输出，查看原始结果
    print("\nRaw result for formatting:")
    print(environmental_data)

    extracted_values = []
    computed_ratios = []
    in_extracted_values = False
    in_computed_ratios = False

    for line in environmental_data.split('\n'):
        title = line.strip()

        if re.search(r'(?i)#?\s*extracted\s+(environmental\s+metrics|values)[:：]?', title):
            in_extracted_values = True
            in_computed_ratios = False
            continue
        elif re.search(r'(?i)#?\s*computed\s+ratios[:：]?', title):  # 如果你的输出中没有这部分，可以删除或调整
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
    print(f"{'Environmental Data Analysis Results':^76}")
    print("=" * 80)
    if extracted_values:
        print("\n" + "-" * 80)
        print(f"{'Extracted Environmental Metrics':^76}")  # 修改为环境指标
        print("-" * 80)
        print(tabulate(extracted_values, headers=["Environmental Metric", "Value"], tablefmt="grid"))  # 修改表头
    else:
        print("\nNo environmental metrics found in the results.")


    if computed_ratios:
        print("\n" + "-" * 80)
        print(f"{'Additional Environmental Information':^76}")  # 修改为更适合环境数据的标题
        print("-" * 80)
        print(tabulate(computed_ratios, headers=["Information", "Value"], tablefmt="grid"))  # 修改表头
    else:
        print("\nNo additional environmental information found in the results.")
    print("=" * 80 + "\n")


extraction_params = {
    "Total Global GHG Emissions": ["Total GHG Emissions", "Operational Emissions", "Scope 1, 2 and 3"],
    "Net Global GHG Emissions": ["Net GHG Emissions", "Market-Based", "Net Emissions"],
    "Total Absolute Financed Emissions": ["Financed Emissions", "Portfolio Emissions", "Absolute Financed Emissions"],
    "Total Premises Energy Use": ["Total Energy Consumption", "Premises Energy Use", "Energy Consumption"],
    "Total Renewable Energy Consumption": ["Renewable Energy Consumption", "Renewable Energy", "% Renewable Energy"],
    "Net Energy Consumption": ["Net Energy Use", "Net Energy", "Total Net Energy"],
    "Total Road Transport Energy Use": ["Transport Energy Use", "Fleet Emissions", "Road Transport"],
    "Total Paper Use": ["Paper Consumption", "Paper Use", "Paper"],
    "Total Water Consumption": ["Total Water Usage", "Water Consumption", "Water Use"],
    "Waste Recycling Rate": ["Waste Recycling Rate", "% Recycled", "Recycling Rate"],
    "Total Sustainable Finance Deals": ["Sustainable Finance", "Green Loans", "Climate-related Lending"],
    "Products Supporting Low-Carbon Transition": ["Green Products", "Climate Products", "Lending to Support Transition"]
}


# 主函数
def main():
    """Main function"""
    api_key = input("Please enter your Anthropic API key: ").strip()
    if not api_key:
        print("Warning: No API key provided. API calls will fail.")

    # Step 1: Get PDF path
    print("\n" + "*" * 50)
    print(f"{'Enhanced Report Analysis Tool':^46}")
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

    # Step 9: Prepare environmental analysis query
    environmental_analysis_query = """
        You are a senior environmental analyst specializing in corporate sustainability reporting. Your task is to carefully extract environmental data from the provided annual report and ensure all metrics are standardized with consistent units. Focus ONLY on the 2024 data.

        ENVIRONMENTAL METRICS TO EXTRACT (for 2024 only):

        1. Total Global GHG Emissions (Scope 1, 2 & 3 - Market-Based)
        2. Net Global GHG Emissions (Scope 1, 2 & 3 - Market-Based) (tonnes CO2e)
        3. Total Absolute Financed Emissions (MtCO₂-e)
        4. Total Premises Energy Use Consumed (MWh)
        5. Total Renewable Energy Consumption (MWh)
        6. Net Energy Consumption (MWh)
        7. Total Road Transport Energy Use (MWh)
        8. Total Paper Use (Tonnes)
        9. Total Water Consumption (KL)
        10. Waste Recycling Rate (%)
        11. Total Sustainable Finance Deals ($m)
        12. Products for Personal and Business Customers Supporting Low-Carbon Transition ($m)

        IMPORTANT:

        1. Be thorough in searching for data across the entire document.
        2. Pay special attention to sections titled: "Environmental Performance", "Sustainability Report", "ESG Highlights", "Climate Change", "Carbon Footprint", "Energy Use", "Waste Management", "Water Management", or "Sustainable Finance".
        3. Recognize that environmental indicators may appear under various synonyms:
            1. Total Global GHG Emissions (Scope 1, 2 & 3 - Market-Based)(tonnes CO2e): Total GHG Emissions (Scope 1, 2 and 3) / Operational Emissions
            2. Net Global GHG Emissions (Scope 1, 2 & 3 - Market-Based) (tonnes CO2e): Net GHG Emissions (Market-Based)
            3. Total Absolute Financed Emissions (MtCO₂-e): Financed Emissions / Portfolio Emissions / Total Absolute Financed Emissions
            4. Total Premises Energy Use Consumed (MWh): Total Energy Consumption / Premises Energy Use
            5. Total Renewable Energy Consumption (MWh): Renewable Energy Consumption / % Renewable Energy
            6. Net Energy Consumption (MWh): Net Energy Use
            7. Total Road Transport Energy Use (MWh): Transport Energy Use / Fleet Emissions
            8. Total Paper Use (tonnes): Paper Consumption
            9.  Total Water Consumption (KL): Total Water Usage / Water Consumption
            10. Waste Recycling Rate (%): Waste Recycling Rate / % Recycled
            11. Total Sustainable Finance Deals ($m): Sustainable Finance / Green Loans / Climate-related Lending
            12. Products for Personal and Business Customers Supporting Low-Carbon Transition ($m): Green/Climate Products for Retail & SME / Lending to Support Transition
        4. STANDARDIZE ALL UNITS AS FOLLOWS:
           - GHG Emissions: Metric tonnes CO2e (CO2 equivalent)
           - Energy Use: MWh (Megawatt hours)
           - Paper Use: Metric tonnes
           - Waste: Metric kiloliter (KL)
           - Water: Cubic meters (m³)
           - Financial Metrics: $m
        5. For each value extracted, include the EXACT sentence where it appears in quotation marks.
        6. If a value cannot be found after thorough examination, explicitly state "Value not found" in place of the value.


        OUTPUT FORMAT:

        Extracted Environmental Metrics:
        Total Global GHG Emissions (Scope 1, 2 & 3 - Market-Based): <value in tonnes CO2e> tonnes CO2e (from "<exact source sentence>")
        Net Global GHG Emissions (Scope 1, 2 & 3 - Market-Based): <value in tonnes CO2e> tonnes CO2e (from "<exact source sentence>")
        Total Absolute Financed Emissions: <value in MtCO₂-e>  MtCO₂-e (from "<exact source sentence>") 
        Total Premises Energy Use Consumed: <value in MWh> MWh (from "<exact source sentence>")
        Total Renewable Energy Consumption: <value in MWh> MWh (from "<exact source sentence>")
        Net Energy Consumption: <value in MWh> MWh (from "<exact source sentence>")
        Total Road Transport Energy Use: <value in MWh> MWh (from "<exact source sentence>")
        Total Paper Use: <value in tonnes> tonnes (from "<exact source sentence>")
        Total Water Consumption: <value in KL> KL (from "<exact source sentence>")
        Waste Recycling Rate: <percentage> % (from "<exact source sentence>")
        Total Sustainable Finance Deals: <value in millions> million (from "<exact source sentence>")
        Products for Personal and Business Customers Supporting Low-Carbon Transition: <value in $m> $m (from "<exact source sentence>")
        """
    # Step 10: Analyze data using Anthropic Claude
    result = analyze_environmental_data(api_key, text_chunks, environmental_analysis_query)
    # Step 11: Format results
    format_environmental_results(result)
    print("\nReport analysis completed successfully!")


if __name__ == "__main__":
    main()