"""
Generate realistic sample real estate PDF documents for the RAG prototype.

Creates intentionally varied and messy documents to demonstrate why
standard RAG tools fail on real-world real estate data:
  - Mixed tables + narrative text
  - Inconsistent formatting across document types
  - Headers/footers that confuse naive parsers
  - Misaligned table columns

Usage:
    python -m data.generate_sample_pdfs
"""

import logging
import random
from pathlib import Path

from fpdf import FPDF

from config import PDF_DIR

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Realistic sample data
# ──────────────────────────────────────────────

PROPERTIES = [
    {
        "address": "742 Evergreen Terrace, Austin, TX 78701",
        "sqft": 2450,
        "bedrooms": 4,
        "bathrooms": 3,
        "lot_size": "0.28 acres",
        "year_built": 2018,
        "assessed_value": 525000,
        "listing_price": 549900,
        "neighborhood": "Downtown Austin",
    },
    {
        "address": "1510 Maple Drive, Austin, TX 78704",
        "sqft": 1850,
        "bedrooms": 3,
        "bathrooms": 2,
        "lot_size": "0.19 acres",
        "year_built": 2005,
        "assessed_value": 415000,
        "listing_price": 439000,
        "neighborhood": "South Congress",
    },
    {
        "address": "405 Congress Ave, Austin, TX 78701",
        "sqft": 1950,
        "bedrooms": 3,
        "bathrooms": 2,
        "lot_size": "0.18 acres",
        "year_built": 2015,
        "assessed_value": 510000,
        "listing_price": 529000,
        "neighborhood": "Downtown Austin",
    },
    {
        "address": "328 Riverside Blvd, Austin, TX 78702",
        "sqft": 3100,
        "bedrooms": 5,
        "bathrooms": 4,
        "lot_size": "0.35 acres",
        "year_built": 2021,
        "assessed_value": 720000,
        "listing_price": 749900,
        "neighborhood": "East Austin",
    },
    {
        "address": "89 Lakewood Circle, Austin, TX 78703",
        "sqft": 2100,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "lot_size": "0.22 acres",
        "year_built": 2012,
        "assessed_value": 485000,
        "listing_price": 499000,
        "neighborhood": "Tarrytown",
    },
    {
        "address": "2204 Bluebonnet Lane, Austin, TX 78745",
        "sqft": 1600,
        "bedrooms": 3,
        "bathrooms": 2,
        "lot_size": "0.15 acres",
        "year_built": 1998,
        "assessed_value": 345000,
        "listing_price": 359900,
        "neighborhood": "South Lamar",
    },
]

COMPARABLE_SALES = [
    {"address": "750 Evergreen Terrace, Austin, TX 78701", "sale_price": 535000, "sale_date": "2024-08-15", "sqft": 2380, "price_per_sqft": 224.79},
    {"address": "718 Evergreen Terrace, Austin, TX 78701", "sale_price": 510000, "sale_date": "2024-06-22", "sqft": 2290, "price_per_sqft": 222.71},
    {"address": "1520 Maple Drive, Austin, TX 78704", "sale_price": 425000, "sale_date": "2024-07-10", "sqft": 1900, "price_per_sqft": 223.68},
    {"address": "1498 Maple Drive, Austin, TX 78704", "sale_price": 398000, "sale_date": "2024-05-30", "sqft": 1780, "price_per_sqft": 223.60},
    {"address": "340 Riverside Blvd, Austin, TX 78702", "sale_price": 705000, "sale_date": "2024-09-01", "sqft": 3050, "price_per_sqft": 231.15},
    {"address": "312 Riverside Blvd, Austin, TX 78702", "sale_price": 690000, "sale_date": "2024-04-18", "sqft": 2980, "price_per_sqft": 231.54},
    {"address": "95 Lakewood Circle, Austin, TX 78703", "sale_price": 492000, "sale_date": "2024-07-25", "sqft": 2150, "price_per_sqft": 228.84},
    {"address": "77 Lakewood Circle, Austin, TX 78703", "sale_price": 478000, "sale_date": "2024-06-05", "sqft": 2050, "price_per_sqft": 233.17},
    {"address": "2210 Bluebonnet Lane, Austin, TX 78745", "sale_price": 352000, "sale_date": "2024-08-20", "sqft": 1650, "price_per_sqft": 213.33},
    {"address": "2198 Bluebonnet Lane, Austin, TX 78745", "sale_price": 340000, "sale_date": "2024-03-12", "sqft": 1580, "price_per_sqft": 215.19},
    {"address": "411 Congress Ave, Austin, TX 78701", "sale_price": 515000, "sale_date": "2024-07-30", "sqft": 1980, "price_per_sqft": 260.10},
    {"address": "399 Congress Ave, Austin, TX 78701", "sale_price": 498000, "sale_date": "2024-05-15", "sqft": 1870, "price_per_sqft": 266.31},
]

NEIGHBORHOODS = {
    "Downtown Austin": {"avg_price": 545000, "median_price": 530000, "avg_dom": 22, "inventory": 145, "trend": "rising", "yoy_change": 4.2},
    "South Congress": {"avg_price": 428000, "median_price": 420000, "avg_dom": 28, "inventory": 112, "trend": "stable", "yoy_change": 1.8},
    "East Austin": {"avg_price": 685000, "median_price": 710000, "avg_dom": 18, "inventory": 88, "trend": "rising", "yoy_change": 6.1},
    "Tarrytown": {"avg_price": 498000, "median_price": 490000, "avg_dom": 25, "inventory": 95, "trend": "stable", "yoy_change": 2.3},
    "South Lamar": {"avg_price": 358000, "median_price": 350000, "avg_dom": 32, "inventory": 168, "trend": "declining", "yoy_change": -1.5},
}


class RealEstatePDF(FPDF):
    """Custom FPDF subclass with real estate document styling."""

    def __init__(self, doc_title: str, include_header_footer: bool = True):
        super().__init__()
        self.doc_title = doc_title
        self.include_header_footer = include_header_footer
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        if self.include_header_footer:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            # Edge case: headers that could confuse parsing
            self.cell(0, 5, f"CONFIDENTIAL - {self.doc_title} - Austin Metro Area MLS", align="C")
            self.ln(8)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)

    def footer(self):
        if self.include_header_footer:
            self.set_y(-20)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 5, f"Page {self.page_no()} | Generated by Austin Real Estate Analytics Platform | {random.choice(['Q3 2024', 'Q4 2024'])}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 60, 120)
        self.cell(0, 10, title)
        self.ln(12)
        self.set_text_color(0, 0, 0)

    def sub_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title)
        self.ln(10)
        self.set_text_color(0, 0, 0)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(4)

    def add_table(self, headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None):
        """Add a table with optional misaligned columns (realistic edge case)."""
        if col_widths is None:
            col_widths = [int(190 / len(headers))] * len(headers)

        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(220, 230, 240)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, border=1, fill=True)
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 9)
        for row in rows:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1)
            self.ln()
        self.ln(5)


def generate_appraisal_report(prop: dict, comps: list[dict]) -> None:
    """Generate a property appraisal report PDF with mixed tables and narrative."""
    pdf = RealEstatePDF("Property Appraisal Report")
    pdf.add_page()

    pdf.section_title("PROPERTY APPRAISAL REPORT")
    pdf.body_text(f"Report Date: October 15, 2024")
    pdf.body_text(f"Appraiser: Jennifer Martinez, MAI, SRA  |  License #TX-APR-2847")
    pdf.ln(3)

    # Property details as mixed narrative + data
    pdf.sub_title("Subject Property Information")
    pdf.body_text(f"The subject property is located at {prop['address']}. "
                  f"This is a single-family residential property built in {prop['year_built']} "
                  f"comprising approximately {prop['sqft']:,} square feet of living space on a "
                  f"{prop['lot_size']} lot. The home features {prop['bedrooms']} bedrooms and "
                  f"{prop['bathrooms']} bathrooms.")
    pdf.ln(2)

    # Property details table
    pdf.sub_title("Property Summary")
    pdf.add_table(
        headers=["Feature", "Detail"],
        rows=[
            ["Address", prop["address"]],
            ["Square Footage", f"{prop['sqft']:,} sq ft"],
            ["Bedrooms", str(prop["bedrooms"])],
            ["Bathrooms", str(prop["bathrooms"])],
            ["Lot Size", prop["lot_size"]],
            ["Year Built", str(prop["year_built"])],
            ["Neighborhood", prop["neighborhood"]],
        ],
        col_widths=[60, 130],
    )

    # Narrative assessment (mixed with data — this is what makes parsing hard)
    pdf.sub_title("Appraiser's Assessment")
    pdf.body_text(
        f"Based on my inspection conducted on October 10, 2024, the subject property at "
        f"{prop['address']} is in good overall condition. The roof was replaced in "
        f"{prop['year_built'] + random.randint(2, 5)} and the HVAC system appears to be "
        f"functioning properly. Interior finishes are consistent with properties in the "
        f"${prop['assessed_value'] - 50000:,} to ${prop['assessed_value'] + 50000:,} range "
        f"in the {prop['neighborhood']} area."
    )
    pdf.body_text(
        f"The kitchen features granite countertops and stainless steel appliances. "
        f"Flooring is a mix of hardwood in common areas and carpet in bedrooms. "
        f"The primary bathroom was recently updated with modern fixtures."
    )

    # Comparable sales table — the key data
    pdf.add_page()
    pdf.section_title("COMPARABLE SALES ANALYSIS")
    pdf.body_text(
        "The following comparable sales were selected based on proximity, recency of sale, "
        "and similarity of property characteristics to the subject property."
    )

    # Table with intentionally tight columns (edge case)
    comp_rows = []
    for c in comps[:4]:
        comp_rows.append([
            c["address"].split(",")[0],  # just street for space
            f"${c['sale_price']:,}",
            c["sale_date"],
            f"{c['sqft']:,}",
            f"${c['price_per_sqft']:.2f}",
        ])

    pdf.add_table(
        headers=["Address", "Sale Price", "Date", "Sq Ft", "$/Sq Ft"],
        rows=comp_rows,
        col_widths=[55, 35, 30, 30, 30],
    )

    # Valuation conclusion
    pdf.sub_title("Valuation Conclusion")
    avg_ppsf = sum(c["price_per_sqft"] for c in comps[:4]) / 4
    estimated = int(avg_ppsf * prop["sqft"])
    pdf.body_text(
        f"Based on the comparable sales analysis, the estimated market value of the "
        f"subject property at {prop['address']} is ${estimated:,}. "
        f"This valuation is based on an average price per square foot of ${avg_ppsf:.2f} "
        f"derived from {len(comps[:4])} comparable sales in the {prop['neighborhood']} area. "
        f"The county assessed value is ${prop['assessed_value']:,}, which is "
        f"{'below' if prop['assessed_value'] < estimated else 'above'} our estimated market value."
    )

    filename = f"appraisal_{prop['address'].split(',')[0].replace(' ', '_').lower()}.pdf"
    filepath = PDF_DIR / filename
    pdf.output(str(filepath))
    logger.info(f"Generated appraisal report: {filename}")


def generate_listing_sheet(prop: dict) -> None:
    """Generate an MLS-style property listing sheet."""
    pdf = RealEstatePDF("MLS Property Listing", include_header_footer=True)
    pdf.add_page()

    # Big header like a real listing
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 80, 160)
    pdf.cell(0, 12, "PROPERTY LISTING", align="C")
    pdf.ln(15)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"MLS# ATX-{random.randint(100000, 999999)}")
    pdf.ln(5)
    pdf.cell(0, 8, f"List Price: ${prop['listing_price']:,}")
    pdf.ln(10)

    # Address block
    pdf.sub_title("Property Address")
    pdf.body_text(prop["address"])

    # Features in a table
    pdf.sub_title("Property Features")
    features = [
        ["Bedrooms", str(prop["bedrooms"]), "Bathrooms", str(prop["bathrooms"])],
        ["Square Feet", f"{prop['sqft']:,}", "Lot Size", prop["lot_size"]],
        ["Year Built", str(prop["year_built"]), "Style", "Single Family"],
        ["Garage", f"{random.choice([2, 3])}-Car Attached", "Heating", "Central"],
        ["Cooling", "Central A/C", "Flooring", "Hardwood/Carpet"],
    ]
    for row in features:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(35, 6, row[0], border=1)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(55, 6, row[1], border=1)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(35, 6, row[2], border=1)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(55, 6, row[3], border=1)
        pdf.ln()
    pdf.ln(8)

    # Agent description — narrative text
    pdf.sub_title("Agent Remarks")
    remarks = [
        f"Beautiful {prop['bedrooms']}-bedroom home in the heart of {prop['neighborhood']}! ",
        f"This stunning {prop['sqft']:,} sq ft property features an open floor plan with "
        f"plenty of natural light throughout. ",
        f"Recently updated kitchen with quartz countertops, custom cabinetry, and premium appliances. ",
        f"The spacious primary suite offers a walk-in closet and spa-like bathroom. ",
        f"Enjoy the private backyard oasis with mature trees and a covered patio, perfect for entertaining. ",
        f"Walking distance to shops, restaurants, and parks. ",
        f"Zoned for top-rated Austin ISD schools. Don't miss this one!",
    ]
    pdf.body_text("".join(remarks))

    # Days on market info
    dom = random.randint(5, 45)
    pdf.body_text(f"Days on Market: {dom}")
    pdf.body_text(f"Listing Date: September {random.randint(1,30)}, 2024")
    pdf.body_text(f"Listed by: {random.choice(['Sarah Johnson', 'Mike Chen', 'Lisa Rodriguez'])}, "
                  f"{random.choice(['Keller Williams Austin', 'RE/MAX Premier', 'Compass Austin'])}")

    filename = f"listing_{prop['address'].split(',')[0].replace(' ', '_').lower()}.pdf"
    filepath = PDF_DIR / filename
    pdf.output(str(filepath))
    logger.info(f"Generated listing sheet: {filename}")


def generate_market_analysis() -> None:
    """Generate a neighborhood market analysis report with statistics and trends."""
    pdf = RealEstatePDF("Market Analysis Report")
    pdf.add_page()

    pdf.section_title("AUSTIN METRO AREA MARKET ANALYSIS")
    pdf.body_text("Report Period: Q3 2024  |  Prepared by: Austin Real Estate Analytics Group")
    pdf.body_text("Data Sources: Austin Board of REALTORS, Travis County Appraisal District, MLS")
    pdf.ln(5)

    # Executive summary — narrative
    pdf.sub_title("Executive Summary")
    pdf.body_text(
        "The Austin metropolitan real estate market continues to show mixed signals across "
        "different neighborhoods in Q3 2024. Downtown and East Austin areas are experiencing "
        "price appreciation driven by tech sector growth and urban development, while outer "
        "suburban areas like South Lamar are seeing slight price corrections after the rapid "
        "appreciation of 2021-2023. Overall market inventory has increased 12% year-over-year, "
        "giving buyers more negotiating power in some segments."
    )

    # Neighborhood stats table
    pdf.sub_title("Neighborhood Performance Summary")
    rows = []
    for name, data in NEIGHBORHOODS.items():
        rows.append([
            name,
            f"${data['avg_price']:,}",
            f"${data['median_price']:,}",
            str(data["avg_dom"]),
            str(data["inventory"]),
            f"{data['yoy_change']:+.1f}%",
        ])

    pdf.add_table(
        headers=["Neighborhood", "Avg Price", "Median Price", "Avg DOM", "Inventory", "YoY Change"],
        rows=rows,
        col_widths=[35, 30, 32, 22, 28, 28],
    )

    # Per-neighborhood narrative analysis
    for name, data in NEIGHBORHOODS.items():
        pdf.sub_title(f"{name} - Detailed Analysis")
        trend_text = {
            "rising": "prices have been steadily increasing",
            "stable": "prices have remained relatively stable",
            "declining": "prices have experienced a slight decline",
        }
        pdf.body_text(
            f"In the {name} neighborhood, {trend_text[data['trend']]} over the past 12 months, "
            f"with a year-over-year change of {data['yoy_change']:+.1f}%. The average sale price "
            f"currently stands at ${data['avg_price']:,} with a median of ${data['median_price']:,}. "
            f"Properties are spending an average of {data['avg_dom']} days on market before going "
            f"under contract, with {data['inventory']} active listings as of Q3 2024."
        )

    filepath = PDF_DIR / "market_analysis_austin_q3_2024.pdf"
    pdf.output(str(filepath))
    logger.info("Generated market analysis report")


def generate_comparable_sales_report() -> None:
    """Generate a comparable sales report with tabular data."""
    pdf = RealEstatePDF("Comparable Sales Report")
    pdf.add_page()

    pdf.section_title("COMPARABLE SALES REPORT")
    pdf.body_text("Austin Metropolitan Area  |  Sales Period: March 2024 - September 2024")
    pdf.body_text("Compiled from Travis County public records and MLS data.")
    pdf.ln(5)

    # Main comp sales table
    pdf.sub_title("Recent Comparable Sales")
    rows = []
    for c in COMPARABLE_SALES:
        rows.append([
            c["address"].split(",")[0],
            f"${c['sale_price']:,}",
            c["sale_date"],
            f"{c['sqft']:,}",
            f"${c['price_per_sqft']:.2f}",
        ])

    pdf.add_table(
        headers=["Address", "Sale Price", "Date", "Sq Ft", "$/Sq Ft"],
        rows=rows,
        col_widths=[50, 35, 30, 28, 28],
    )

    # Summary statistics — narrative mixed with numbers (hard for naive parsers)
    pdf.sub_title("Statistical Summary")
    prices = [c["sale_price"] for c in COMPARABLE_SALES]
    ppsf = [c["price_per_sqft"] for c in COMPARABLE_SALES]
    pdf.body_text(
        f"Across the {len(COMPARABLE_SALES)} comparable sales analyzed, the average sale price "
        f"was ${sum(prices)/len(prices):,.0f} with a range from ${min(prices):,} to "
        f"${max(prices):,}. The average price per square foot was ${sum(ppsf)/len(ppsf):.2f}, "
        f"indicating relatively consistent pricing across the sample. "
        f"The most recent sale ({COMPARABLE_SALES[4]['sale_date']}) at "
        f"{COMPARABLE_SALES[4]['address'].split(',')[0]} closed at "
        f"${COMPARABLE_SALES[4]['sale_price']:,}, suggesting continued demand in the area."
    )

    filepath = PDF_DIR / "comparable_sales_austin_2024.pdf"
    pdf.output(str(filepath))
    logger.info("Generated comparable sales report")


def generate_all_pdfs() -> list[Path]:
    """Generate all sample PDFs and return the list of file paths."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    generated = []

    # Appraisal reports for first 3 properties (with matching comps)
    for i, prop in enumerate(PROPERTIES[:3]):
        # Pick 2-4 comps relevant to this property's street or neighborhood
        street_name = prop["address"].split(",")[0].split()[-1].lower()  # e.g., "terrace", "drive", "ave"
        relevant_comps = [c for c in COMPARABLE_SALES if street_name in c["address"].lower()]
        if not relevant_comps:
            # Fallback: grab comps by index
            relevant_comps = COMPARABLE_SALES[i*2:(i*2)+3]
        generate_appraisal_report(prop, relevant_comps)
        generated.append(PDF_DIR / f"appraisal_{prop['address'].split(',')[0].replace(' ', '_').lower()}.pdf")

    # Listing sheets for all properties
    for prop in PROPERTIES:
        generate_listing_sheet(prop)
        generated.append(PDF_DIR / f"listing_{prop['address'].split(',')[0].replace(' ', '_').lower()}.pdf")

    # Market analysis
    generate_market_analysis()
    generated.append(PDF_DIR / "market_analysis_austin_q3_2024.pdf")

    # Comparable sales
    generate_comparable_sales_report()
    generated.append(PDF_DIR / "comparable_sales_austin_2024.pdf")

    logger.info(f"Generated {len(generated)} sample PDFs in {PDF_DIR}")
    return generated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = generate_all_pdfs()
    print(f"\nGenerated {len(paths)} PDFs:")
    for p in paths:
        print(f"  {p.name}")
