from langchain_chroma import Chroma
from langchain_core.documents import Document

from embeddings import embeddings_model

technical_docs = [
    Document(
        page_content="Error E1: Indicates overheating of the main pressure pump. To resolve: turn off the machinery for at least 15 minutes, check that the blue relief valve on the back is fully open, then restart the unit.",
        metadata={"source": "pump_manual_v2"}
    ),
    Document(
        page_content="Error E2: Lack of water in the hydraulic circuit. Check the connection of the gray inlet hose and ensure that the upstream valve is open.",
        metadata={"source": "hidraulic_manual_v1"}
    ),
    Document(
        page_content="Air Filter Cleaning: The air filter must be removed by gently rotating it counterclockwise. Wash it under lukewarm running water without detergents. Dry completely before reinserting.",
        metadata={"source": "generic_manual"}
    )
]

reference_emails = [
    Document(
        page_content="""
Dear Customer,
Thank you for contacting us. We are sincerely sorry for the inconvenience you have experienced with our device.
Regarding the technical issue you reported, we kindly ask you to follow this procedure:
- Check that the cable is correctly positioned.
- Press the reset button on the back for 5 seconds.
- Wait for the system to fully reboot before proceeding.
Should you still encounter any difficulties, please feel free to reply to this message and one of our operators will assist you.
Best regards,
The Technical Support Team
        """,
        metadata={"intent": "troubleshooting", "style": "empathetic_formal"}
    ),
    Document(
        page_content="""
Dear Customer,
Thank you for requesting information about our products.
In regard to your inquiry, I can confirm that routine maintenance of the outer casings requires the exclusive use of a dry microfiber cloth. Please strictly avoid abrasive sprays or chemical solvents, as they could dull the plastic surfaces.
We remain at your complete disposal for any further needs.
Best regards,
Centralized Customer Service
        """,
        metadata={"intent": "maintenance_information", "style": "informative_courteous"}
    )
]

tech_db = Chroma.from_documents(
    documents=technical_docs,
    embedding=embeddings_model
)
technical_retriever = tech_db.as_retriever(search_kwargs={"k": 1})

style_db = Chroma.from_documents(
    documents=reference_emails,
    embedding=embeddings_model
)
style_retriever = style_db.as_retriever(search_kwargs={"k": 1})
