"""
Comprehensive tests for vitalDSP_webapp.layout.common components.
Tests footer, header, and sidebar components.
"""

import pytest
from dash import html
import dash_bootstrap_components as dbc

# Import the modules we need to test
try:
    from vitalDSP_webapp.layout.common.footer import Footer
    from vitalDSP_webapp.layout.common.header import Header
    from vitalDSP_webapp.layout.common.sidebar import Sidebar
except ImportError:
    # Fallback: add src to path if import fails
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..', '..')
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from vitalDSP_webapp.layout.common.footer import Footer
    from vitalDSP_webapp.layout.common.header import Header
    from vitalDSP_webapp.layout.common.sidebar import Sidebar


class TestFooter:
    """Test Footer component functionality"""
    
    def test_footer_creation(self):
        """Test Footer component can be created"""
        footer = Footer()
        
        assert footer is not None
        # Dash components have different attribute structure
        assert hasattr(footer, '__class__')
        assert footer.__class__.__name__ == 'Footer'
        
    def test_footer_structure(self):
        """Test Footer component structure"""
        footer = Footer()
        
        assert footer.className == "footer"
        assert hasattr(footer, 'style')
        assert hasattr(footer, 'children')
        
    def test_footer_styling(self):
        """Test Footer component styling"""
        footer = Footer()
        
        style = footer.style
        assert style['position'] == 'fixed'
        assert style['bottom'] == '0'
        assert style['left'] == '0'
        assert style['right'] == '0'
        assert style['height'] == '40px'
        assert style['backgroundColor'] == '#2c3e50'
        assert style['color'] == 'white'
        assert style['zIndex'] == 1000
        assert style['display'] == 'flex'
        assert style['alignItems'] == 'center'
        assert style['justifyContent'] == 'center'
        assert style['fontSize'] == '12px'
        assert style['opacity'] == '0.8'
        
    def test_footer_content(self):
        """Test Footer component content"""
        footer = Footer()
        
        assert len(footer.children) == 1
        span_element = footer.children[0]
        assert span_element.__class__.__name__ == 'Span'
        assert "© 2024 vitalDSP" in span_element.children
        assert "Digital Signal Processing for Vital Signs" in span_element.children
        
    def test_footer_accessibility(self):
        """Test Footer component accessibility features"""
        footer = Footer()
        
        # Test that footer is properly structured for screen readers
        assert footer.__class__.__name__ == 'Footer'
        assert footer.children[0].__class__.__name__ == 'Span'


class TestHeader:
    """Test Header component functionality"""
    
    def test_header_creation(self):
        """Test Header component can be created"""
        header = Header()
        
        assert header is not None
        assert hasattr(header, '__class__')
        assert header.__class__.__name__ == 'Header'
        
    def test_header_structure(self):
        """Test Header component structure"""
        header = Header()
        
        assert header.className == "header"
        assert hasattr(header, 'style')
        assert hasattr(header, 'children')
        # Header now has 5 children: title div, status div, memory store, system store, interval
        assert len(header.children) == 5
        
    def test_header_styling(self):
        """Test Header component styling"""
        header = Header()
        
        style = header.style
        assert style['position'] == 'fixed'
        assert style['top'] == '0'
        assert style['left'] == '0'
        assert style['right'] == '0'
        assert style['height'] == '60px'
        assert style['backgroundColor'] == '#2c3e50'
        assert style['color'] == 'white'
        assert style['zIndex'] == 1000
        assert style['display'] == 'flex'
        assert style['alignItems'] == 'center'
        assert style['padding'] == '0 20px'
        assert style['boxShadow'] == '0 2px 4px rgba(0,0,0,0.1)'
        
    def test_header_title(self):
        """Test Header component title element"""
        header = Header()
        
        # Title is now in the first child (Div) which contains H1
        title_div = header.children[0]
        assert title_div.__class__.__name__ == 'Div'
        # The H1 is inside the title div
        title_element = title_div.children[0]
        assert title_element.__class__.__name__ == 'H1'
        assert title_element.children == "vitalDSP"
        assert title_element.className == "mb-0"
        
        title_style = title_element.style
        assert title_style['fontSize'] == '24px'
        assert title_style['fontWeight'] == 'bold'
        assert title_style['color'] == 'white'
        
    def test_header_subtitle(self):
        """Test Header component subtitle element"""
        header = Header()
        
        # Subtitle is now in the first child (Div) which contains Span
        title_div = header.children[0]
        assert title_div.__class__.__name__ == 'Div'
        # The Span is inside the title div
        subtitle_element = title_div.children[1]
        assert subtitle_element.__class__.__name__ == 'Span'
        assert subtitle_element.children == "Digital Signal Processing for Vital Signs"
        assert subtitle_element.className == "ms-3"
        
        subtitle_style = subtitle_element.style
        assert subtitle_style['fontSize'] == '14px'
        assert subtitle_style['opacity'] == '0.8'
        
    def test_header_accessibility(self):
        """Test Header component accessibility features"""
        header = Header()
        
        # Test semantic HTML structure
        assert header.__class__.__name__ == 'Header'
        # Title is now in the first child (Div) which contains H1
        title_div = header.children[0]
        assert title_div.__class__.__name__ == 'Div'
        # The H1 is inside the title div
        title_element = title_div.children[0]
        assert title_element.__class__.__name__ == 'H1'  # Main title
        # Subtitle is also in the first child (Div) which contains Span
        subtitle_element = title_div.children[1]
        assert subtitle_element.__class__.__name__ == 'Span'  # Subtitle


class TestSidebar:
    """Test Sidebar component functionality"""
    
    def test_sidebar_creation(self):
        """Test Sidebar component can be created"""
        sidebar = Sidebar()
        
        assert sidebar is not None
        assert hasattr(sidebar, '__class__')
        assert sidebar.__class__.__name__ == 'Div'
        
    def test_sidebar_structure(self):
        """Test Sidebar component structure"""
        sidebar = Sidebar()
        
        assert sidebar.id == "sidebar"
        assert "sidebar" in sidebar.className
        assert "sidebar-expanded" in sidebar.className
        assert hasattr(sidebar, 'children')
        assert len(sidebar.children) == 3  # toggle button, content, icons
        
    def test_sidebar_toggle_button(self):
        """Test Sidebar toggle button"""
        sidebar = Sidebar()
        
        toggle_button = sidebar.children[0]
        assert toggle_button.__class__.__name__ == 'Button'
        assert toggle_button.id == "sidebar-toggle"
        assert "sidebar-toggle" in toggle_button.className
        assert "btn" in toggle_button.className
        assert "btn-link" in toggle_button.className
        assert "text-white" in toggle_button.className
        
        # Test toggle icon
        assert len(toggle_button.children) == 1
        icon = toggle_button.children[0]
        assert icon.__class__.__name__ == 'I'
        assert icon.id == "sidebar-toggle-icon"
        assert "fas fa-bars" in icon.className
        
    def test_sidebar_content_structure(self):
        """Test Sidebar content structure"""
        sidebar = Sidebar()
        
        content = sidebar.children[1]
        assert content.__class__.__name__ == 'Div'
        assert "sidebar-content" in content.className
        assert len(content.children) > 5  # Should have navigation title and sections
        
    def test_sidebar_navigation_sections(self):
        """Test Sidebar navigation sections"""
        sidebar = Sidebar()
        content = sidebar.children[1]
        
        # Find navigation title
        nav_title = content.children[0]
        assert nav_title.__class__.__name__ == 'H5'
        assert nav_title.children == "Navigation"
        assert "mb-3" in nav_title.className
        
        # Test that upload section exists
        upload_section = content.children[1]
        assert upload_section.__class__.__name__ == 'Div'
        assert "mb-2" in upload_section.className
        
    def test_sidebar_upload_link(self):
        """Test Sidebar upload link"""
        sidebar = Sidebar()
        content = sidebar.children[1]
        upload_section = content.children[1]
        
        upload_link = upload_section.children[0]
        assert upload_link.__class__.__name__ == 'A'
        assert upload_link.href == "/upload"
        assert "nav-link" in upload_link.className
        assert "text-white" in upload_link.className
        
        # Test link content
        assert len(upload_link.children) == 2
        icon = upload_link.children[0]
        assert icon.__class__.__name__ == 'I'
        assert "fas fa-upload" in icon.className
        assert upload_link.children[1] == " Upload Data"
        
    def test_sidebar_analysis_section(self):
        """Test Sidebar analysis section"""
        sidebar = Sidebar()
        content = sidebar.children[1]
        
        # Find analysis section header
        analysis_header = None
        analysis_links = None
        
        for i, child in enumerate(content.children):
            if hasattr(child, 'children') and child.children == "Analysis":
                analysis_header = child
                analysis_links = content.children[i + 1]
                break
                
        assert analysis_header is not None
        assert analysis_header.__class__.__name__ == 'H6'
        assert "text-muted" in analysis_header.className
        
        assert analysis_links is not None
        assert analysis_links.__class__.__name__ == 'Div'
        assert len(analysis_links.children) >= 3  # Time domain, frequency, filtering
        
    def test_sidebar_analysis_links(self):
        """Test Sidebar analysis links"""
        sidebar = Sidebar()
        content = sidebar.children[1]
        
        # Find analysis links section
        analysis_links = None
        for child in content.children:
            if (hasattr(child, 'children') and 
                len(child.children) >= 3 and
                hasattr(child.children[0], 'href') and
                child.children[0].href == "/filtering"):
                analysis_links = child
                break
                
        assert analysis_links is not None
        
        # Test filtering link (now first)
        filtering_link = analysis_links.children[0]
        assert filtering_link.href == "/filtering"
        assert "nav-link" in filtering_link.className
        
        # Test time domain link (now second)
        time_link = analysis_links.children[1]
        assert time_link.href == "/time-domain"
        assert "nav-link" in time_link.className
        assert "text-white" in time_link.className
        
        # Test frequency link (now at index 2)
        freq_link = analysis_links.children[2]
        assert freq_link.href == "/frequency"
        
    def test_sidebar_features_section(self):
        """Test Sidebar features section"""
        sidebar = Sidebar()
        content = sidebar.children[1]
        
        # Find features section
        features_links = None
        for child in content.children:
            if (hasattr(child, 'children') and 
                len(child.children) >= 3 and
                hasattr(child.children[0], 'href') and
                child.children[0].href == "/physiological"):
                features_links = child
                break
                
        assert features_links is not None
        
        # Test physiological link
        physio_link = features_links.children[0]
        assert physio_link.href == "/physiological"
        
        # Test respiratory link
        resp_link = features_links.children[1]
        assert resp_link.href == "/respiratory"
        
        # Test advanced features link
        adv_link = features_links.children[2]
        assert adv_link.href == "/features"
        
    def test_sidebar_other_section(self):
        """Test Sidebar other section"""
        sidebar = Sidebar()
        content = sidebar.children[1]
        
        # Find other section
        other_links = None
        for child in content.children:
            if (hasattr(child, 'children') and 
                len(child.children) >= 2 and
                hasattr(child.children[0], 'href') and
                child.children[0].href == "/preview"):
                other_links = child
                break
                
        assert other_links is not None
        
        # Test preview link
        preview_link = other_links.children[0]
        assert preview_link.href == "/preview"
        
        # Test settings link
        settings_link = other_links.children[1]
        assert settings_link.href == "/settings"
        
    def test_sidebar_icons_section(self):
        """Test Sidebar icons section (collapsed view)"""
        sidebar = Sidebar()
        
        icons_section = sidebar.children[2]
        assert icons_section.__class__.__name__ == 'Div'
        assert "sidebar-icons" in icons_section.className
        assert len(icons_section.children) >= 4  # Different icon groups
        
    def test_sidebar_icon_links(self):
        """Test Sidebar icon links structure"""
        sidebar = Sidebar()
        icons_section = sidebar.children[2]
        
        # Test upload icon
        upload_icon_section = icons_section.children[0]
        upload_icon_link = upload_icon_section.children[0]
        assert upload_icon_link.__class__.__name__ == 'A'
        assert upload_icon_link.href == "/upload"
        assert "nav-icon" in upload_icon_link.className
        assert "text-white" in upload_icon_link.className
        assert upload_icon_link.title == "Upload Data"
        
        # Test icon element
        icon = upload_icon_link.children
        assert icon.__class__.__name__ == 'I'
        assert "fas fa-upload" in icon.className
        
    def test_sidebar_responsive_design(self):
        """Test Sidebar responsive design elements"""
        sidebar = Sidebar()
        
        # Test that both expanded and collapsed views exist
        content_section = sidebar.children[1]  # Expanded view
        icons_section = sidebar.children[2]    # Collapsed view
        
        assert "sidebar-content" in content_section.className
        assert "sidebar-icons" in icons_section.className
        
        # Test that icon links have tooltips for collapsed view
        icon_link = icons_section.children[0].children[0]
        assert hasattr(icon_link, 'title')
        assert icon_link.title is not None
        
    def test_sidebar_accessibility(self):
        """Test Sidebar accessibility features"""
        sidebar = Sidebar()
        
        # Test semantic structure
        assert sidebar.__class__.__name__ == 'Div'
        assert sidebar.id == "sidebar"
        
        # Test that links have proper href attributes
        content = sidebar.children[1]
        for child in content.children:
            if hasattr(child, 'children'):
                for subchild in child.children:
                    if hasattr(subchild, '__class__') and subchild.__class__.__name__ == 'A':
                        assert hasattr(subchild, 'href')
                        assert subchild.href.startswith('/')
                        
        # Test that icon links have title attributes for tooltips
        icons_section = sidebar.children[2]
        for icon_group in icons_section.children:
            for icon_link in icon_group.children:
                if hasattr(icon_link, '__class__') and icon_link.__class__.__name__ == 'A':
                    assert hasattr(icon_link, 'title')
                    assert icon_link.title is not None


class TestLayoutIntegration:
    """Test layout components integration"""
    
    def test_all_components_can_be_created(self):
        """Test that all layout components can be created together"""
        header = Header()
        sidebar = Sidebar()
        footer = Footer()
        
        assert header is not None
        assert sidebar is not None
        assert footer is not None
        
    def test_components_have_consistent_styling(self):
        """Test that components have consistent styling"""
        header = Header()
        footer = Footer()
        
        # Test consistent color scheme
        assert header.style['backgroundColor'] == footer.style['backgroundColor']
        assert header.style['color'] == footer.style['color']
        
    def test_components_z_index_layering(self):
        """Test that components have proper z-index layering"""
        header = Header()
        footer = Footer()
        
        # Both should have same z-index for proper layering
        assert header.style['zIndex'] == 1000
        assert footer.style['zIndex'] == 1000
        
    def test_layout_positioning(self):
        """Test that layout components are positioned correctly"""
        header = Header()
        footer = Footer()
        
        # Header at top
        assert header.style['position'] == 'fixed'
        assert header.style['top'] == '0'
        
        # Footer at bottom
        assert footer.style['position'] == 'fixed'
        assert footer.style['bottom'] == '0'
        
    def test_sidebar_navigation_completeness(self):
        """Test that sidebar contains all expected navigation items"""
        sidebar = Sidebar()
        
        # Extract all href values from sidebar
        hrefs = set()
        
        def extract_hrefs(element):
            if hasattr(element, 'href') and element.href:
                hrefs.add(element.href)
            if hasattr(element, 'children'):
                if isinstance(element.children, list):
                    for child in element.children:
                        extract_hrefs(child)
                elif hasattr(element.children, 'href'):
                    extract_hrefs(element.children)
                    
        extract_hrefs(sidebar)
        
        # Test that all expected pages are linked
        expected_pages = {
            "/upload", "/time-domain", "/frequency", "/filtering",
            "/physiological", "/respiratory", "/features",
            "/preview", "/settings"
        }
        
        for page in expected_pages:
            assert page in hrefs, f"Expected page {page} not found in sidebar navigation"
