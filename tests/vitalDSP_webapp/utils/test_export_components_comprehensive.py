"""
Comprehensive tests for export_components.py to improve coverage.

This file adds extensive coverage for export UI component functions.
"""

import pytest
from dash import html, dcc
import dash_bootstrap_components as dbc
from vitalDSP_webapp.utils.export_components import (
    create_export_buttons,
    create_export_card,
    create_inline_export_buttons,
    create_dropdown_export_menu,
    create_export_section_with_preview,
)


class TestCreateExportButtons:
    """Test create_export_buttons function."""

    def test_create_export_buttons_basic(self):
        """Test basic export buttons creation."""
        result = create_export_buttons("test_page")
        assert isinstance(result, dbc.Row)
        assert len(result.children) > 0

    def test_create_export_buttons_custom_text(self):
        """Test export buttons with custom text."""
        result = create_export_buttons("test_page", button_text="Custom Export")
        assert isinstance(result, dbc.Row)

    def test_create_export_buttons_structure(self):
        """Test export buttons structure contains buttons and download components."""
        result = create_export_buttons("test_page")
        # Check that it contains Col with ButtonGroup
        col = result.children[0]
        assert isinstance(col, dbc.Col)
        # Check for download components (they should be in the children)
        assert len(col.children) >= 2  # ButtonGroup + Download components


class TestCreateExportCard:
    """Test create_export_card function."""

    def test_create_export_card_basic(self):
        """Test basic export card creation."""
        result = create_export_card("test_page")
        assert isinstance(result, dbc.Card)
        assert len(result.children) >= 2  # CardHeader + CardBody

    def test_create_export_card_custom_title(self):
        """Test export card with custom title."""
        result = create_export_card("test_page", title="Custom Title")
        assert isinstance(result, dbc.Card)

    def test_create_export_card_structure(self):
        """Test export card structure."""
        result = create_export_card("test_page")
        # Should have CardHeader and CardBody
        assert hasattr(result, 'children')
        assert len(result.children) >= 2


class TestCreateInlineExportButtons:
    """Test create_inline_export_buttons function."""

    def test_create_inline_export_buttons_basic(self):
        """Test basic inline export buttons creation."""
        result = create_inline_export_buttons("test_page")
        assert isinstance(result, html.Div)
        assert len(result.children) >= 2  # ButtonGroup + Download components

    def test_create_inline_export_buttons_structure(self):
        """Test inline export buttons structure."""
        result = create_inline_export_buttons("test_page")
        # Should contain ButtonGroup and Download components
        assert len(result.children) >= 2


class TestCreateDropdownExportMenu:
    """Test create_dropdown_export_menu function."""

    def test_create_dropdown_export_menu_basic(self):
        """Test basic dropdown export menu creation."""
        result = create_dropdown_export_menu("test_page")
        assert isinstance(result, html.Div)
        assert len(result.children) >= 2  # DropdownMenu + Download components

    def test_create_dropdown_export_menu_structure(self):
        """Test dropdown export menu structure."""
        result = create_dropdown_export_menu("test_page")
        # Should contain DropdownMenu and Download components
        assert len(result.children) >= 2


class TestCreateExportSectionWithPreview:
    """Test create_export_section_with_preview function."""

    def test_create_export_section_with_preview_basic(self):
        """Test basic export section with preview creation."""
        result = create_export_section_with_preview("test_page")
        assert isinstance(result, dbc.Card)
        assert len(result.children) >= 2  # CardHeader + CardBody

    def test_create_export_section_with_preview_custom_title(self):
        """Test export section with custom title."""
        result = create_export_section_with_preview("test_page", title="Custom Export")
        assert isinstance(result, dbc.Card)

    def test_create_export_section_with_preview_structure(self):
        """Test export section structure."""
        result = create_export_section_with_preview("test_page")
        # Should have CardHeader and CardBody
        assert hasattr(result, 'children')
        assert len(result.children) >= 2


class TestExportComponentsEdgeCases:
    """Test edge cases for export components."""

    def test_create_export_buttons_different_page_ids(self):
        """Test export buttons with different page IDs."""
        page_ids = ["filtered", "time-domain", "frequency", "quality"]
        for page_id in page_ids:
            result = create_export_buttons(page_id)
            assert isinstance(result, dbc.Row)

    def test_create_export_card_different_page_ids(self):
        """Test export card with different page IDs."""
        page_ids = ["filtered", "time-domain", "frequency", "quality"]
        for page_id in page_ids:
            result = create_export_card(page_id)
            assert isinstance(result, dbc.Card)

    def test_create_inline_export_buttons_different_page_ids(self):
        """Test inline export buttons with different page IDs."""
        page_ids = ["filtered", "time-domain", "frequency", "quality"]
        for page_id in page_ids:
            result = create_inline_export_buttons(page_id)
            assert isinstance(result, html.Div)

    def test_create_dropdown_export_menu_different_page_ids(self):
        """Test dropdown export menu with different page IDs."""
        page_ids = ["filtered", "time-domain", "frequency", "quality"]
        for page_id in page_ids:
            result = create_dropdown_export_menu(page_id)
            assert isinstance(result, html.Div)

    def test_create_export_section_with_preview_different_page_ids(self):
        """Test export section with different page IDs."""
        page_ids = ["filtered", "time-domain", "frequency", "quality"]
        for page_id in page_ids:
            result = create_export_section_with_preview(page_id)
            assert isinstance(result, dbc.Card)

