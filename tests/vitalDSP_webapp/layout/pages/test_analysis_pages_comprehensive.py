"""
Comprehensive tests for vitalDSP_webapp.layout.pages.analysis_pages module.
Tests all analysis page layouts and their components.
"""

import pytest
from dash import html, dcc
import dash_bootstrap_components as dbc

# Import the modules we need to test
try:
    from vitalDSP_webapp.layout.pages.analysis_pages import (
        time_domain_layout,
        frequency_layout,
        filtering_layout,
        physiological_layout,
        respiratory_layout,
        features_layout,
        transforms_layout,
        quality_layout,
        advanced_layout,
        health_report_layout,
        settings_layout
    )
except ImportError:
    # Fallback: add src to path if import fails
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..', '..')
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from vitalDSP_webapp.layout.pages.analysis_pages import (
        time_domain_layout,
        frequency_layout,
        filtering_layout,
        physiological_layout,
        respiratory_layout,
        features_layout,
        transforms_layout,
        quality_layout,
        advanced_layout,
        health_report_layout,
        settings_layout
    )


class TestTimeDomainLayout:
    """Test time domain analysis page layout"""
    
    def test_time_domain_layout_creation(self):
        """Test time domain layout can be created"""
        layout = time_domain_layout()
        
        assert layout is not None
        assert hasattr(layout, '__class__')
        assert layout.__class__.__name__ == 'Div'
        
    def test_time_domain_layout_structure(self):
        """Test time domain layout structure"""
        layout = time_domain_layout()
        
        assert hasattr(layout, 'children')
        assert len(layout.children) > 0
        
        # Should have header section
        header_section = layout.children[0]
        assert header_section.__class__.__name__ == 'Div'
        
    def test_time_domain_layout_header(self):
        """Test time domain layout header"""
        layout = time_domain_layout()
        header_section = layout.children[0]
        
        # Should contain title
        title_element = None
        for child in header_section.children:
            if hasattr(child, '__class__') and child.__class__.__name__ == 'H1':
                title_element = child
                break
                
        assert title_element is not None
        assert "Time Domain Analysis" in title_element.children
        
    def test_time_domain_layout_buttons(self):
        """Test time domain layout action buttons"""
        layout = time_domain_layout()
        
        # Find button section - it's a dbc.Row, not html.Div
        button_section = None
        for child in layout.children:
            if (hasattr(child, '__class__') and 
                (child.__class__.__name__ == 'Row' or 'Row' in str(child.__class__))):
                button_section = child
                break
                
        assert button_section is not None, f"Expected to find Row component but found: {[child.__class__.__name__ for child in layout.children]}"


class TestFrequencyLayout:
    """Test frequency domain analysis page layout"""
    
    def test_frequency_layout_creation(self):
        """Test frequency layout can be created"""
        layout = frequency_layout()
        
        assert layout is not None
        assert hasattr(layout, '__class__')
        assert layout.__class__.__name__ == 'Div'
        
    def test_frequency_layout_structure(self):
        """Test frequency layout has proper structure"""
        layout = frequency_layout()
        
        assert hasattr(layout, 'children')
        assert len(layout.children) > 0


class TestFilteringLayout:
    """Test filtering analysis page layout"""
    
    def test_filtering_layout_creation(self):
        """Test filtering layout can be created"""
        layout = filtering_layout()
        
        assert layout is not None
        assert hasattr(layout, '__class__')
        assert layout.__class__.__name__ == 'Div'
        
    def test_filtering_layout_structure(self):
        """Test filtering layout has proper structure"""
        layout = filtering_layout()
        
        assert hasattr(layout, 'children')
        assert len(layout.children) > 0


class TestPhysiologicalLayout:
    """Test physiological features page layout"""
    
    def test_physiological_layout_creation(self):
        """Test physiological layout can be created"""
        layout = physiological_layout()
        
        assert layout is not None
        assert hasattr(layout, '__class__')
        assert layout.__class__.__name__ == 'Div'
        
    def test_physiological_layout_structure(self):
        """Test physiological layout has proper structure"""
        layout = physiological_layout()
        
        assert hasattr(layout, 'children')
        assert len(layout.children) > 0


class TestRespiratoryLayout:
    """Test respiratory analysis page layout"""
    
    def test_respiratory_layout_creation(self):
        """Test respiratory layout can be created"""
        layout = respiratory_layout()
        
        assert layout is not None
        assert hasattr(layout, '__class__')
        assert layout.__class__.__name__ == 'Div'
        
    def test_respiratory_layout_structure(self):
        """Test respiratory layout has proper structure"""
        layout = respiratory_layout()
        
        assert hasattr(layout, 'children')
        assert len(layout.children) > 0


class TestFeaturesLayout:
    """Test features extraction page layout"""
    
    def test_features_layout_creation(self):
        """Test features layout can be created"""
        layout = features_layout()
        
        assert layout is not None
        assert hasattr(layout, '__class__')
        assert layout.__class__.__name__ == 'Div'
        
    def test_features_layout_structure(self):
        """Test features layout has proper structure"""
        layout = features_layout()
        
        assert hasattr(layout, 'children')
        assert len(layout.children) > 0


class TestTransformsLayout:
    """Test transforms analysis page layout"""
    
    def test_transforms_layout_creation(self):
        """Test transforms layout can be created"""
        layout = transforms_layout()
        
        assert layout is not None
        assert hasattr(layout, '__class__')
        assert layout.__class__.__name__ == 'Div'
        
    def test_transforms_layout_structure(self):
        """Test transforms layout has proper structure"""
        layout = transforms_layout()
        
        assert hasattr(layout, 'children')
        assert len(layout.children) > 0


class TestQualityLayout:
    """Test quality assessment page layout"""
    
    def test_quality_layout_creation(self):
        """Test quality layout can be created"""
        layout = quality_layout()
        
        assert layout is not None
        assert hasattr(layout, '__class__')
        assert layout.__class__.__name__ == 'Div'
        
    def test_quality_layout_structure(self):
        """Test quality layout has proper structure"""
        layout = quality_layout()
        
        assert hasattr(layout, 'children')
        assert len(layout.children) > 0


class TestAdvancedLayout:
    """Test advanced analysis page layout"""
    
    def test_advanced_layout_creation(self):
        """Test advanced layout can be created"""
        layout = advanced_layout()
        
        assert layout is not None
        assert hasattr(layout, '__class__')
        assert layout.__class__.__name__ == 'Div'
        
    def test_advanced_layout_structure(self):
        """Test advanced layout has proper structure"""
        layout = advanced_layout()
        
        assert hasattr(layout, 'children')
        assert len(layout.children) > 0


class TestHealthReportLayout:
    """Test health report page layout"""
    
    def test_health_report_layout_creation(self):
        """Test health report layout can be created"""
        layout = health_report_layout()
        
        assert layout is not None
        assert hasattr(layout, '__class__')
        assert layout.__class__.__name__ == 'Div'
        
    def test_health_report_layout_structure(self):
        """Test health report layout has proper structure"""
        layout = health_report_layout()
        
        assert hasattr(layout, 'children')
        assert len(layout.children) > 0


class TestSettingsLayout:
    """Test settings page layout"""
    
    def test_settings_layout_creation(self):
        """Test settings layout can be created"""
        layout = settings_layout()
        
        assert layout is not None
        assert hasattr(layout, '__class__')
        assert layout.__class__.__name__ == 'Div'
        
    def test_settings_layout_structure(self):
        """Test settings layout has proper structure"""
        layout = settings_layout()
        
        assert hasattr(layout, 'children')
        assert len(layout.children) > 0


class TestLayoutsIntegration:
    """Test analysis pages integration"""
    
    def test_all_layouts_can_be_created(self):
        """Test that all analysis page layouts can be created"""
        layouts = [
            time_domain_layout(),
            frequency_layout(),
            filtering_layout(),
            physiological_layout(),
            respiratory_layout(),
            features_layout(),
            transforms_layout(),
            quality_layout(),
            advanced_layout(),
            health_report_layout(),
            settings_layout()
        ]
        
        for layout in layouts:
            assert layout is not None
            # Dash components don't have a 'tag' attribute, they have different attributes
            assert hasattr(layout, '__class__')
            assert layout.__class__.__name__ == 'Div'
            
    def test_layouts_have_content(self):
        """Test that all layouts have content"""
        layouts = [
            time_domain_layout(),
            frequency_layout(),
            filtering_layout(),
            physiological_layout(),
            respiratory_layout(),
            features_layout(),
            transforms_layout(),
            quality_layout(),
            advanced_layout(),
            health_report_layout(),
            settings_layout()
        ]
        
        for layout in layouts:
            assert hasattr(layout, 'children')
            assert len(layout.children) > 0
            
    def test_layouts_are_different(self):
        """Test that layouts are different from each other"""
        layouts = [
            time_domain_layout(),
            frequency_layout(),
            filtering_layout(),
            physiological_layout(),
            respiratory_layout(),
            features_layout(),
            transforms_layout(),
            quality_layout(),
            advanced_layout(),
            health_report_layout(),
            settings_layout()
        ]
        
        # Convert layouts to strings for comparison (basic check)
        layout_strings = [str(layout) for layout in layouts]
        
        # Check that they're not all identical
        unique_layouts = set(layout_strings)
        assert len(unique_layouts) > 1  # Should have different layouts
        
    def test_common_layout_elements(self):
        """Test common elements across layouts"""
        layouts = [
            time_domain_layout(),
            frequency_layout(),
            filtering_layout(),
            physiological_layout(),
            respiratory_layout()
        ]
        
        for layout in layouts:
            # Each layout should be a div
            assert layout.__class__.__name__ == 'Div'
            
            # Each layout should have children
            assert hasattr(layout, 'children')
            assert isinstance(layout.children, list)
            assert len(layout.children) > 0
            
    def test_layout_accessibility(self):
        """Test layout accessibility features"""
        layout = time_domain_layout()
        
        # Should use semantic HTML elements
        def check_semantic_elements(element):
            semantic_tags = {'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'Header', 'Main', 'Section', 'Article'}
            found_semantic = False
            
            if hasattr(element, '__class__') and element.__class__.__name__ in semantic_tags:
                found_semantic = True
                
            if hasattr(element, 'children') and isinstance(element.children, list):
                for child in element.children:
                    if check_semantic_elements(child):
                        found_semantic = True
                        
            return found_semantic
            
        # At least one layout should use semantic elements
        assert check_semantic_elements(layout)
        
    def test_responsive_design_elements(self):
        """Test responsive design elements in layouts"""
        layout = time_domain_layout()
        
        # Should use Bootstrap responsive classes
        def check_responsive_classes(element):
            responsive_found = False
            
            if hasattr(element, 'className'):
                class_str = str(element.className) if element.className else ""
                # Check for Bootstrap responsive classes
                responsive_classes = ['col-', 'row', 'container', 'card', 'mb-', 'mt-', 'p-', 'm-']
                if any(cls in class_str for cls in responsive_classes):
                    responsive_found = True
                    
            if hasattr(element, 'children') and isinstance(element.children, list):
                for child in element.children:
                    if check_responsive_classes(child):
                        responsive_found = True
                        
            return responsive_found
            
        assert check_responsive_classes(layout)
        
    def test_interactive_elements(self):
        """Test interactive elements in layouts"""
        layout = time_domain_layout()
        
        # Should contain interactive elements like buttons, inputs, etc.
        def check_interactive_elements(element):
            interactive_found = False
            interactive_tags = {'Button', 'Input', 'Select', 'Textarea'}
            
            if hasattr(element, '__class__') and element.__class__.__name__ in interactive_tags:
                interactive_found = True
                
            # Check for Dash components
            if hasattr(element, '__class__'):
                class_name = element.__class__.__name__
                if any(comp in class_name for comp in ['Button', 'Input', 'Dropdown', 'Select']):
                    interactive_found = True
                    
            if hasattr(element, 'children') and isinstance(element.children, list):
                for child in element.children:
                    if check_interactive_elements(child):
                        interactive_found = True
                        
            return interactive_found
            
        assert check_interactive_elements(layout)


class TestLayoutPerformance:
    """Test layout performance considerations"""
    
    def test_layout_creation_performance(self):
        """Test that layouts can be created efficiently"""
        import time
        
        start_time = time.time()
        
        # Create all layouts
        layouts = [
            time_domain_layout(),
            frequency_layout(),
            filtering_layout(),
            physiological_layout(),
            respiratory_layout(),
            features_layout(),
            transforms_layout(),
            quality_layout(),
            advanced_layout(),
            health_report_layout(),
            settings_layout()
        ]
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Should create all layouts within reasonable time (5 seconds)
        assert creation_time < 5.0
        assert len(layouts) == 11
        
    def test_layout_memory_usage(self):
        """Test that layouts don't consume excessive memory"""
        import sys
        
        # Get initial memory usage
        initial_size = sys.getsizeof(None)
        
        # Create a layout
        layout = time_domain_layout()
        
        # Check that layout size is reasonable (less than 10MB)
        layout_size = sys.getsizeof(layout)
        assert layout_size < 10 * 1024 * 1024  # 10MB
        
    def test_layout_component_count(self):
        """Test that layouts don't have excessive component counts"""
        layout = time_domain_layout()
        
        def count_components(element):
            count = 1
            if hasattr(element, 'children') and isinstance(element.children, list):
                for child in element.children:
                    count += count_components(child)
            return count
            
        component_count = count_components(layout)
        
        # Should have reasonable number of components (less than 1000)
        assert component_count < 1000
