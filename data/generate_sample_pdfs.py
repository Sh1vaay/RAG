"""
Generates sample company documents (PDFs) used as source material for the RAG pipeline.
Mirrors the internship project's real source docs: service brochures, SOPs, case studies.
Two of these documents (FTTH and Electrical guides) deliberately share overlapping
vocabulary ("installation", "business days", "technician") to reproduce the real
retrieval-confusion bug described in the project write-up.
"""

import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT_DIR = os.path.join(os.path.dirname(__file__), "docs")
os.makedirs(OUT_DIR, exist_ok=True)

styles = getSampleStyleSheet()


def make_pdf(filename, title, category, paragraphs):
    path = os.path.join(OUT_DIR, filename)
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=0.8 * inch, bottomMargin=0.8 * inch)
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    story.append(Paragraph(f"Category: {category}", styles["Heading3"]))
    story.append(Spacer(1, 12))
    for p in paragraphs:
        story.append(Paragraph(p, styles["BodyText"]))
        story.append(Spacer(1, 10))
    doc.build(story)
    print(f"wrote {path}")


make_pdf(
    "cctv_amc_brochure.pdf",
    "CCTV Surveillance & AMC Services Brochure",
    "CCTV",
    [
        "Our CCTV surveillance division offers end-to-end camera installation, monitoring, "
        "and Annual Maintenance Contract (AMC) services for commercial and residential clients.",
        "The standard AMC plan includes quarterly on-site inspection, firmware updates across "
        "all connected devices, and complimentary repair coverage for up to two camera units per "
        "calendar year. Repairs beyond this allowance are billed at standard technician rates.",
        "For AMC clients, our service level agreement guarantees an on-site technician response "
        "within 24 hours for critical outages such as full system downtime, and within 72 hours "
        "for non-critical issues such as a single malfunctioning camera unit.",
        "24/7 remote monitoring is available as a separately billed add-on to any installation "
        "package and includes real-time alerting for motion events and camera tampering.",
        "New installations are available in 2MP, 4MP, and 8MP (4K) resolution options, with 4MP "
        "being the most commonly deployed resolution for commercial retail and office sites.",
    ],
)

make_pdf(
    "ftth_installation_guide.pdf",
    "FTTH (Fiber-to-the-Home) Installation Guide",
    "FTTH",
    [
        "Fiber-to-the-Home (FTTH) installation brings dedicated fiber-optic connectivity "
        "directly to residential and commercial premises, supporting symmetric internet plans "
        "from 100 Mbps up to 1 Gbps.",
        "A standard residential FTTH installation takes 3 to 5 business days from the initial "
        "site survey through to final activation, assuming no permitting or right-of-way delays "
        "are encountered during the trenching phase.",
        "Before installation begins, our technician will require proof of premises ownership or "
        "a valid tenancy agreement, a signed service contract, and local right-of-way clearance "
        "documentation if underground trenching is required for the fiber run.",
        "Installation in multi-unit apartment buildings follows a similar process but may extend "
        "the overall timeline to 7 to 10 business days, depending on how quickly building "
        "management approves shared-conduit access for the installation crew.",
        "Once activated, our support desk can assist with router configuration and initial "
        "speed testing to confirm the installed plan is performing as expected.",
    ],
)

make_pdf(
    "electrical_installation_guide.pdf",
    "Electrical & Solar Installation Guide",
    "Electrical",
    [
        "Our electrical installation division handles wiring, panel upgrades, and safety "
        "certification for commercial and residential clients, often bundled with our solar "
        "installation services.",
        "A standard electrical installation for a small commercial unit takes approximately 5 to "
        "7 business days from the initial technician site visit through final safety testing and "
        "certification of the completed panel work.",
        "When electrical and solar installation are bundled together, the combined project "
        "timeline typically extends to 10 to 14 business days end-to-end, covering wiring, panel "
        "setup, solar mounting, inverter installation, and final grid connection approval.",
        "Every completed electrical installation includes a mandatory safety inspection "
        "certificate issued by our certifying technician after final testing, which clients will "
        "need for regulatory compliance and insurance purposes.",
        "Residential solar-only installations, when not bundled with broader electrical work, "
        "typically take 4 to 6 business days including mounting, wiring, and grid connection "
        "approval from the local utility provider.",
    ],
)

make_pdf(
    "wifi_solutions_brochure.pdf",
    "Wi-Fi & Network Solutions Brochure",
    "WiFi",
    [
        "We design and deploy Wi-Fi networking solutions for offices, warehouses, and "
        "multi-floor commercial buildings, including industrial-grade access point placement "
        "for large open floor plans such as warehouses and distribution centers.",
        "A small office Wi-Fi setup for up to 20 concurrent users typically takes 1 to 2 "
        "business days, including router and access point configuration and a final coverage "
        "walkthrough with the client.",
        "For spaces larger than 3,000 square feet or spanning multiple floors, we recommend and "
        "support mesh Wi-Fi network configurations to eliminate dead zones and provide seamless "
        "roaming between access points.",
        "Warehouse Wi-Fi deployments typically require a dedicated site survey to account for "
        "high-density shelving, metal racking interference, and the need for reliable coverage "
        "across large open floor areas.",
    ],
)

make_pdf(
    "smart_city_case_study.pdf",
    "Smart City Pilot Program Case Study",
    "SmartCity",
    [
        "Our Smart City division has delivered pilot programs combining smart lighting, traffic "
        "sensor deployment, and centralized monitoring dashboards for municipal zones seeking to "
        "modernize public infrastructure.",
        "A typical Smart City pilot includes deployment of smart streetlights with adaptive "
        "brightness control, IoT-based traffic flow sensors at key intersections, and a "
        "centralized dashboard for real-time monitoring by city staff.",
        "Pilot programs typically run for 6 to 12 months, followed by a formal review period "
        "during which outcome metrics are evaluated before any decision is made regarding "
        "city-wide expansion of the piloted technologies.",
        "Structured cabling for Smart City control cabinets is certified against TIA/EIA "
        "standards, with full test reports provided to the municipal client upon project "
        "completion.",
    ],
)

make_pdf(
    "finops_control_overview.pdf",
    "FinOps Control Platform Overview",
    "Cloud",
    [
        "FinOps Control is our flagship cloud cost optimization platform, currently supporting "
        "AWS and Azure, providing unified cost visibility, real-time usage insights, and "
        "automated savings recommendations for client organizations.",
        "Clients typically see up to a 40% reduction in monthly cloud spend after onboarding, "
        "driven primarily by automated right-sizing recommendations and elimination of idle or "
        "orphaned cloud resources identified by the platform.",
        "Standard onboarding takes 3 to 5 business days, covering cloud account connection, an "
        "initial cost audit across all connected accounts, and configuration of the client's "
        "cost visibility dashboard.",
        "The platform emphasizes fast return on investment and continuous support, with a "
        "dedicated FinOps analyst available to clients during the first 90 days post-onboarding.",
    ],
)

make_pdf(
    "bpo_service_sop.pdf",
    "BPO Service Desk Standard Operating Procedure",
    "BPO",
    [
        "Our BPO service desk handles inbound support requests across voice, email, and chat "
        "channels, with standard priority tickets resolved within 24 business hours and urgent "
        "priority tickets carrying a 4-hour resolution target.",
        "Support is currently available in English and Hindi, with regional language add-ons "
        "available for enterprise clients requiring broader language coverage across their "
        "customer base.",
        "Tickets that cannot be resolved at first contact are escalated via the client portal or "
        "by email to our escalations team, which reviews and responds to escalated tickets "
        "within 4 business hours of receipt.",
        "Quality assurance reviews are conducted weekly across a sample of resolved tickets to "
        "ensure resolution quality and adherence to client-specific service scripts and "
        "compliance requirements.",
    ],
)

make_pdf(
    "datacenter_colocation_brochure.pdf",
    "Data Center Co-location Services Brochure",
    "DataCenter",
    [
        "Our data center offers quarter-rack, half-rack, and full-rack co-location options, "
        "with custom cage configurations available for enterprise clients requiring dedicated "
        "physical security zones.",
        "All co-located infrastructure is backed by a 99.9% uptime service level agreement, "
        "with service credits automatically calculated and applied for any downtime beyond the "
        "agreed threshold in a given billing cycle.",
        "Backup power runs on an N+1 redundant configuration combining UPS battery systems with "
        "diesel generator failover, tested monthly as part of standard data center operations.",
        "For clients relocating existing infrastructure, we provide full-service data center "
        "migration including planning, physical relocation, and post-migration validation "
        "testing to confirm all systems are operating correctly at the new location.",
    ],
)

make_pdf(
    "it_infrastructure_amc_guide.pdf",
    "IT Infrastructure AMC Service Guide",
    "ITInfra",
    [
        "Our IT Infrastructure Annual Maintenance Contract covers server hardware monitoring, "
        "quarterly preventive maintenance visits, and priority handling for all support tickets "
        "raised by AMC clients.",
        "24/7 network monitoring with automated alerting is included in all AMC tiers above the "
        "basic plan, allowing our technical team to detect and respond to infrastructure issues "
        "before clients notice any impact.",
        "On-site infrastructure support is available for AMC clients located within a 50 km "
        "radius of our regional service centers, with remote support available to all clients "
        "regardless of location.",
        "We generally recommend a 4 to 5 year hardware refresh cycle for servers and core "
        "networking equipment under an active AMC, based on manufacturer warranty windows and "
        "typical failure-rate curves for enterprise hardware.",
    ],
)

make_pdf(
    "structured_cabling_certification_guide.pdf",
    "Structured Cabling & Certification Guide",
    "StructuredCabling",
    [
        "We install and certify Cat6, Cat6a, and fiber-optic structured cabling for commercial "
        "and industrial clients, with cable category selected based on required bandwidth and "
        "run distance.",
        "Every structured cabling installation is certified against TIA/EIA standards using "
        "calibrated test equipment, with full test reports provided to the client upon project "
        "completion for their own compliance records.",
        "Structured cabling projects for Smart City control cabinets follow the same "
        "certification process, ensuring consistent quality standards across both commercial "
        "office deployments and municipal infrastructure projects.",
    ],
)

make_pdf(
    "solar_maintenance_guide.pdf",
    "Solar Panel Maintenance Guide",
    "Solar",
    [
        "Our solar maintenance plans include quarterly panel cleaning, inverter health checks, "
        "and annual performance efficiency reporting to help clients track system output over "
        "the life of their installation.",
        "New solar panel installations carry a 10-year workmanship warranty covering our "
        "installation work, in addition to the 25-year manufacturer performance warranty "
        "provided on the panels themselves.",
        "We handle the full net metering application process with the local utility provider as "
        "part of every new residential solar installation, so clients don't need to navigate "
        "utility paperwork independently.",
        "A standard electrical installation for a small commercial unit bundled with solar work "
        "takes approximately 10 to 14 business days end-to-end, covering wiring, panel setup, "
        "and final grid connection approval.",
    ],
)

make_pdf(
    "access_control_security_brochure.pdf",
    "Access Control & Security Systems Brochure",
    "AccessControl",
    [
        "We install fingerprint and facial recognition biometric access control systems for "
        "offices, data centers, and secured facility entrances requiring strict identity "
        "verification before entry.",
        "A standard access card system installation includes card reader hardware, access "
        "control software setup, and up to 50 pre-programmed access cards for initial staff "
        "onboarding.",
        "Access control installation for a single-entrance office setup typically takes 2 to 3 "
        "business days, including hardware mounting, software configuration, and a final access "
        "test with facility staff before handover.",
        "Access control systems can be integrated with our CCTV surveillance offering for "
        "combined entry logging and video verification at secured entrances.",
    ],
)

print("All sample PDFs generated.")
