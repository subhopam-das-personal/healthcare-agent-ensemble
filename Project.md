# Healthcare Agent Ensemble

## Overview

This project develops an ensemble of AI agents designed to assist in healthcare applications. The system leverages the Model Context Protocol (MCP) for agent communication and integrates with clinical data sources like FHIR and RxNav.

## Architecture

- **MCP Server**: Hosts healthcare-specific tools for data retrieval and analysis
- **A2A Agent**: Manages agent orchestration and inter-agent communication
- **Shared Clients**: Interfaces to FHIR (patient data), RxNav (drug information), and Claude AI (reasoning)

## Features

- Clinical decision support through multi-agent collaboration
- Drug interaction checking and safety alerts
- Patient data analysis and insights
- Modular design for easy extension with new agents

## Technologies

- Python
- Model Context Protocol (MCP)
- FHIR standards
- RxNav API
- Claude AI integration