import re
from typing import TextIO, List, Optional, Pattern, Match, Dict
from collections import defaultdict
class DebugLogProcessor:
    """Process debug logs to extract and clean specific information blocks with question-based grouping."""
    def __init__(self):
        # Compile regex patterns for better performance
        self.patterns = {
            'prediction': re.compile(r'^Processing Prediction (\d+) for Question (\d+)'),
            'debug_info': re.compile(r'^Debug Information:'),
            'text_processed': re.compile(r'^Text being processed:'),
            'generated_statements': re.compile(r'^Generated statements:'),
            'sentence_index': re.compile(r'^- sentence_index=\d+ simpler_statements=\[(.+)\]')
        }
    def process_statement_list(self, statement_list: str) -> List[str]:
        """Clean and split a list of statements from the log."""
        statements = statement_list.strip("[]").split(", ")
        return [stmt.strip("'\"") for stmt in statements]
    def write_statements(self, statements: List[str]) -> List[str]:
        """Convert statements to output format."""
        return [f'"{statement}"\n' for statement in statements]
    def process_logs(self, input_file: str, output_file: str) -> None:
        """
        Process debug logs and extract relevant information, grouped by question number.
        Args:
            input_file: Path to input log file
            output_file: Path to output cleaned log file
        """
        # First pass: Collect and organize data
        question_data = self._collect_data(input_file)
        # Second pass: Write organized data
        self._write_organized_data(question_data, output_file)
        print("Processing complete! Output file has been generated with grouped questions.")
    def _collect_data(self, input_file: str) -> Dict[int, Dict[int, List[str]]]:
        """
        First pass: Collect all data organized by question and prediction numbers.
        Returns a nested dictionary: {question_number: {prediction_number: [content_lines]}}
        """
        question_data = defaultdict(lambda: defaultdict(list))
        with open(input_file, 'r') as infile:
            lines = infile.readlines()
            current_content = []
            current_question = None
            current_prediction = None
            state = {
                'in_debug_block': False,
                'first_debug_block': False,
                'skip_text_block': False
            }
            for i, line in enumerate(lines):
                # Check for new prediction block
                if pred_match := self.patterns['prediction'].match(line):
                    # Store previous content if it exists
                    if current_question is not None and current_prediction is not None and current_content:
                        question_data[current_question][current_prediction].extend(current_content)
                    # Update current identifiers
                    current_prediction = int(pred_match.group(1))
                    current_question = int(pred_match.group(2))
                    current_content = [line]
                    state['in_debug_block'] = False
                    state['first_debug_block'] = False
                    state['skip_text_block'] = False
                    continue
                # Process other lines based on patterns and state
                if self._should_process_line(line, state):
                    if self.patterns['sentence_index'].match(line):
                        statements = self.process_statement_list(
                            self.patterns['sentence_index'].match(line).group(1)
                        )
                        current_content.extend(self.write_statements(statements))
                    else:
                        current_content.append(line)
            # Don't forget to store the last block
            if current_question is not None and current_prediction is not None and current_content:
                question_data[current_question][current_prediction].extend(current_content)
        return question_data
    def _should_process_line(self, line: str, state: dict) -> bool:
        """Determine if the current line should be processed based on patterns and state."""
        if self.patterns['debug_info'].match(line):
            if not state['first_debug_block']:
                state['first_debug_block'] = True
                state['in_debug_block'] = True
                return True
            else:
                state['in_debug_block'] = False
                return False
        if self.patterns['text_processed'].match(line):
            state['skip_text_block'] = True
            return False
        if self.patterns['generated_statements'].match(line):
            state['skip_text_block'] = False
            return True
        return state['first_debug_block'] and state['in_debug_block'] and not state['skip_text_block']
    def _write_organized_data(self, question_data: Dict[int, Dict[int, List[str]]], output_file: str) -> None:
        """
        Second pass: Write the organized data to the output file.
        Data is written grouped by question number, with predictions in numerical order.
        """
        with open(output_file, 'w') as outfile:
            # Process questions in numerical order
            for question_num in sorted(question_data.keys()):
                # Process predictions for each question in numerical order
                for prediction_num in sorted(question_data[question_num].keys()):
                    content = question_data[question_num][prediction_num]
                    outfile.writelines(content)
def main():
    # Define the paths to input and output files
    input_file = 'terminal_output.log'  # Replace with your input log file
    output_file = 'cleaned_output.log'  # Replace with your desired output log file
    # Create the DebugLogProcessor instance and process the logs
    processor = DebugLogProcessor()
    processor.process_logs(input_file, output_file)

if __name__ == "__main__":
    main()