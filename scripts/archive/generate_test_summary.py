"""
Generate comprehensive test summary report from JUnit XML results.

This script parses test result XML files and creates:
- HTML dashboard with test status
- JSON summary for programmatic access
- Pass/fail statistics
- Execution time metrics
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

# Snakemake inputs/outputs
unit_results = snakemake.input.unit_results
integration_results = snakemake.input.integration_results
html_output = snakemake.output.html
json_output = snakemake.output.json


def parse_junit_xml(xml_file):
    """Parse JUnit XML file and extract test results."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        results = {
            'file': Path(xml_file).name,
            'tests': int(root.get('tests', 0)),
            'failures': int(root.get('failures', 0)),
            'errors': int(root.get('errors', 0)),
            'skipped': int(root.get('skipped', 0)),
            'time': float(root.get('time', 0.0)),
            'test_cases': []
        }
        
        # Parse individual test cases
        for testcase in root.findall('.//testcase'):
            case = {
                'name': testcase.get('name'),
                'classname': testcase.get('classname'),
                'time': float(testcase.get('time', 0.0)),
                'status': 'passed'
            }
            
            # Check for failures
            failure = testcase.find('failure')
            if failure is not None:
                case['status'] = 'failed'
                case['failure_message'] = failure.get('message', '')
                case['failure_text'] = failure.text or ''
            
            # Check for errors
            error = testcase.find('error')
            if error is not None:
                case['status'] = 'error'
                case['error_message'] = error.get('message', '')
                case['error_text'] = error.text or ''
            
            # Check for skipped
            skipped = testcase.find('skipped')
            if skipped is not None:
                case['status'] = 'skipped'
                case['skip_message'] = skipped.get('message', '')
            
            results['test_cases'].append(case)
        
        return results
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return None


def generate_summary():
    """Generate summary statistics from all test results."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'unit_tests': {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'time': 0.0,
            'results': []
        },
        'integration_tests': {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'time': 0.0,
            'results': []
        }
    }
    
    # Parse unit test results
    for xml_file in unit_results:
        if Path(xml_file).exists():
            result = parse_junit_xml(xml_file)
            if result:
                summary['unit_tests']['results'].append(result)
                summary['unit_tests']['total'] += result['tests']
                summary['unit_tests']['failed'] += result['failures']
                summary['unit_tests']['errors'] += result['errors']
                summary['unit_tests']['skipped'] += result['skipped']
                summary['unit_tests']['time'] += result['time']
                summary['unit_tests']['passed'] += (
                    result['tests'] - result['failures'] - result['errors'] - result['skipped']
                )
    
    # Parse integration test results
    for xml_file in integration_results:
        if Path(xml_file).exists():
            result = parse_junit_xml(xml_file)
            if result:
                summary['integration_tests']['results'].append(result)
                summary['integration_tests']['total'] += result['tests']
                summary['integration_tests']['failed'] += result['failures']
                summary['integration_tests']['errors'] += result['errors']
                summary['integration_tests']['skipped'] += result['skipped']
                summary['integration_tests']['time'] += result['time']
                summary['integration_tests']['passed'] += (
                    result['tests'] - result['failures'] - result['errors'] - result['skipped']
                )
    
    return summary


def generate_html(summary):
    """Generate HTML report from summary."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PyPSA-GB Test Summary</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary-box {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .passed {{ color: #4CAF50; }}
        .failed {{ color: #f44336; }}
        .skipped {{ color: #FF9800; }}
        .error {{ color: #d32f2f; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .status-badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .status-passed {{
            background: #4CAF50;
            color: white;
        }}
        .status-failed {{
            background: #f44336;
            color: white;
        }}
        .status-skipped {{
            background: #FF9800;
            color: white;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <h1>🧪 PyPSA-GB Test Results</h1>
    
    <div class="timestamp">Generated: {summary['timestamp']}</div>
    
    <div class="summary-box">
        <h2>📊 Overall Summary</h2>
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Total Tests</div>
                <div class="stat-value">{summary['unit_tests']['total'] + summary['integration_tests']['total']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Passed</div>
                <div class="stat-value passed">{summary['unit_tests']['passed'] + summary['integration_tests']['passed']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Failed</div>
                <div class="stat-value failed">{summary['unit_tests']['failed'] + summary['integration_tests']['failed']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Errors</div>
                <div class="stat-value error">{summary['unit_tests']['errors'] + summary['integration_tests']['errors']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Skipped</div>
                <div class="stat-value skipped">{summary['unit_tests']['skipped'] + summary['integration_tests']['skipped']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Time</div>
                <div class="stat-value">{summary['unit_tests']['time'] + summary['integration_tests']['time']:.2f}s</div>
            </div>
        </div>
    </div>
    
    <div class="summary-box">
        <h2>⚡ Unit Tests</h2>
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Total</div>
                <div class="stat-value">{summary['unit_tests']['total']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Passed</div>
                <div class="stat-value passed">{summary['unit_tests']['passed']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Failed</div>
                <div class="stat-value failed">{summary['unit_tests']['failed']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Time</div>
                <div class="stat-value">{summary['unit_tests']['time']:.2f}s</div>
            </div>
        </div>
        
        <table>
            <tr>
                <th>Test File</th>
                <th>Tests</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Time (s)</th>
                <th>Status</th>
            </tr>
"""
    
    for result in summary['unit_tests']['results']:
        passed = result['tests'] - result['failures'] - result['errors'] - result['skipped']
        status = 'passed' if result['failures'] == 0 and result['errors'] == 0 else 'failed'
        html += f"""
            <tr>
                <td>{result['file']}</td>
                <td>{result['tests']}</td>
                <td class="passed">{passed}</td>
                <td class="failed">{result['failures'] + result['errors']}</td>
                <td>{result['time']:.2f}</td>
                <td><span class="status-badge status-{status}">{status.upper()}</span></td>
            </tr>
"""
    
    html += """
        </table>
    </div>
    
    <div class="summary-box">
        <h2>🔗 Integration Tests</h2>
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Total</div>
                <div class="stat-value">{}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Passed</div>
                <div class="stat-value passed">{}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Failed</div>
                <div class="stat-value failed">{}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Time</div>
                <div class="stat-value">{:.2f}s</div>
            </div>
        </div>
        
        <table>
            <tr>
                <th>Test File</th>
                <th>Tests</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Time (s)</th>
                <th>Status</th>
            </tr>
""".format(
        summary['integration_tests']['total'],
        summary['integration_tests']['passed'],
        summary['integration_tests']['failed'],
        summary['integration_tests']['time']
    )
    
    for result in summary['integration_tests']['results']:
        passed = result['tests'] - result['failures'] - result['errors'] - result['skipped']
        status = 'passed' if result['failures'] == 0 and result['errors'] == 0 else 'failed'
        html += f"""
            <tr>
                <td>{result['file']}</td>
                <td>{result['tests']}</td>
                <td class="passed">{passed}</td>
                <td class="failed">{result['failures'] + result['errors']}</td>
                <td>{result['time']:.2f}</td>
                <td><span class="status-badge status-{status}">{status.upper()}</span></td>
            </tr>
"""
    
    html += """
        </table>
    </div>
    
</body>
</html>
"""
    
    return html


# Generate summary
summary = generate_summary()

# Write JSON output
with open(json_output, 'w') as f:
    json.dump(summary, f, indent=2)

# Write HTML output
html = generate_html(summary)
with open(html_output, 'w') as f:
    f.write(html)

print(f"Test summary generated:")
print(f"  HTML: {html_output}")
print(f"  JSON: {json_output}")
print(f"  Total tests: {summary['unit_tests']['total'] + summary['integration_tests']['total']}")
print(f"  Passed: {summary['unit_tests']['passed'] + summary['integration_tests']['passed']}")
print(f"  Failed: {summary['unit_tests']['failed'] + summary['integration_tests']['failed']}")

