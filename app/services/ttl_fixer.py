"""
TTL (Turtle) Syntax Fixer
Fixes common syntax errors in RDF Turtle files
"""

import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTLSyntaxFixer:
    """Fix common TTL syntax errors"""
    
    def __init__(self):
        self.fixes_applied = []
    
    def fix_file(self, input_path: Path, output_path: Path = None) -> bool:
        """Fix TTL file and save to output path"""
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_fixed.ttl"
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Original file size: {len(content)} characters")
            
            # Apply fixes
            fixed_content = self.apply_fixes(content)
            
            # Write fixed content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            logger.info(f"Fixed file saved to: {output_path}")
            logger.info(f"Fixes applied: {len(self.fixes_applied)}")
            for fix in self.fixes_applied:
                logger.info(f"  - {fix}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fixing file: {e}")
            return False
    
    def apply_fixes(self, content: str) -> str:
        """Apply all syntax fixes"""
        
        # Fix 1: Unclosed strings
        content = self.fix_unclosed_strings(content)
        
        # Fix 2: Missing semicolons or periods
        content = self.fix_missing_terminators(content)
        
        # Fix 3: Invalid escape sequences
        content = self.fix_escape_sequences(content)
        
        # Fix 4: Malformed literals
        content = self.fix_malformed_literals(content)
        
        # Fix 5: Missing closing brackets
        content = self.fix_brackets(content)
        
        # Fix 6: Fix collaborative typo specifically
        content = self.fix_specific_errors(content)
        
        return content
    
    def fix_unclosed_strings(self, content: str) -> str:
        """Fix unclosed string literals"""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Count quotes
            single_quotes = line.count("'") - line.count("\\'")
            double_quotes = line.count('"') - line.count('\\"')
            
            # If odd number of quotes and line doesn't end with terminator
            if (single_quotes % 2 != 0 or double_quotes % 2 != 0):
                # Check if it's missing a closing quote before a terminator
                if ';' in line or '.' in line:
                    # Try to close before the terminator
                    if single_quotes % 2 != 0:
                        line = re.sub(r"(\s+)(;|\.)", r"'\2", line)
                        self.fixes_applied.append(f"Line {i+1}: Added closing single quote")
                    elif double_quotes % 2 != 0:
                        line = re.sub(r'(\s+)(;|\.)', r'"\2', line)
                        self.fixes_applied.append(f"Line {i+1}: Added closing double quote")
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_missing_terminators(self, content: str) -> str:
        """Fix missing semicolons or periods"""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip comments, empty lines, prefixes
            if not stripped or stripped.startswith('#') or stripped.startswith('@'):
                fixed_lines.append(line)
                continue
            
            # Check if line should have terminator
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                
                # If next line starts with a predicate or new subject
                if (next_line and 
                    not next_line.startswith('#') and
                    not next_line.startswith('@') and
                    not stripped.endswith(';') and
                    not stripped.endswith('.') and
                    not stripped.endswith(',') and
                    'currkg:' in next_line):
                    
                    # Check if it's end of subject (needs period) or predicate (needs semicolon)
                    if re.match(r'^\s*currkg:\w+\s+a\s+currkg:', next_line):
                        # New subject - needs period
                        line = line.rstrip() + ' .'
                        self.fixes_applied.append(f"Line {i+1}: Added period")
                    elif re.match(r'^\s*currkg:\w+\s+currkg:', next_line):
                        # New predicate of same subject - needs semicolon
                        line = line.rstrip() + ' ;'
                        self.fixes_applied.append(f"Line {i+1}: Added semicolon")
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_escape_sequences(self, content: str) -> str:
        """Fix invalid escape sequences"""
        # Fix common issues
        fixes = [
            (r'\\n(?!")', r'\\\\n'),  # Literal \n should be \\n
            (r'\\t(?!")', r'\\\\t'),  # Literal \t should be \\t
        ]
        
        for pattern, replacement in fixes:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                self.fixes_applied.append(f"Fixed escape sequences: {pattern}")
        
        return content
    
    def fix_malformed_literals(self, content: str) -> str:
        """Fix malformed string literals"""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Fix: string literal without closing quote before predicate
            # Pattern: "text currkg:something
            if re.search(r'"[^"]*\s+currkg:', line):
                line = re.sub(r'"([^"]*)\s+(currkg:\w+)', r'"\1" ;\n    \2', line)
                self.fixes_applied.append(f"Line {i+1}: Fixed malformed literal")
            
            # Fix: 'text currkg:something
            if re.search(r"'[^']*\s+currkg:", line):
                line = re.sub(r"'([^']*)\s+(currkg:\w+)", r"'\1' ;\n    \2", line)
                self.fixes_applied.append(f"Line {i+1}: Fixed malformed literal")
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_brackets(self, content: str) -> str:
        """Fix missing closing brackets"""
        # Count brackets
        open_square = content.count('[')
        close_square = content.count(']')
        open_paren = content.count('(')
        close_paren = content.count(')')
        
        if open_square > close_square:
            content += '\n' + ']' * (open_square - close_square) + ' .\n'
            self.fixes_applied.append(f"Added {open_square - close_square} closing square brackets")
        
        if open_paren > close_paren:
            content += '\n' + ')' * (open_paren - close_paren) + ' .\n'
            self.fixes_applied.append(f"Added {open_paren - close_paren} closing parentheses")
        
        return content
    
    def fix_specific_errors(self, content: str) -> str:
        """Fix specific known errors"""
        
        # Fix the "llaborative" typo and unclosed string
        if 'llaborative' in content:
            # Pattern: llaborative ;\n    currkg:studentRole currkg:Active'..."
            content = re.sub(
                r'llaborative\s*;\s*currkg:studentRole\s+currkg:Active[\'"]',
                'Collaborative" ;\n    currkg:studentRole currkg:Active',
                content
            )
            self.fixes_applied.append("Fixed 'llaborative' typo and string closure")
        
        # Fix any remaining "co llaborative" patterns
        content = re.sub(r'\bco\s+llaborative\b', 'Collaborative', content, flags=re.IGNORECASE)
        
        # Fix patterns like: 'text..." (mixed quotes)
        content = re.sub(r"'([^']*?)\"", r'"\1"', content)
        content = re.sub(r"\"([^\"]*?)'", r'"\1"', content)
        
        return content
    
    def validate_fixed_file(self, file_path: Path) -> bool:
        """Try to validate the fixed TTL file"""
        try:
            from rdflib import Graph
            g = Graph()
            g.parse(file_path, format='turtle')
            logger.info(f"✓ Validation successful! Loaded {len(g)} triples")
            return True
        except Exception as e:
            logger.error(f"✗ Validation failed: {e}")
            return False


def main():
    """Main function to fix TTL files"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ttl_fixer.py <input_file.ttl> [output_file.ttl]")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    fixer = TTLSyntaxFixer()
    
    # Fix the file
    if fixer.fix_file(input_file, output_file):
        output = output_file or input_file.parent / f"{input_file.stem}_fixed.ttl"
        
        # Validate
        print("\nValidating fixed file...")
        if fixer.validate_fixed_file(output):
            print(f"\n✓ SUCCESS! Fixed file is valid: {output}")
        else:
            print(f"\n✗ File fixed but validation failed. Manual review needed: {output}")
    else:
        print("\n✗ Failed to fix file")
        sys.exit(1)


if __name__ == "__main__":
    main()