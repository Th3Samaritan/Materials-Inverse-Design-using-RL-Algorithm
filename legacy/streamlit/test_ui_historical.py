"""Streamlit interaction checks; use the project virtual environment."""
import unittest
import torch
from pathlib import Path
from streamlit.testing.v1 import AppTest


class UITests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'steel_heat_treatment_recommender.py'), default_timeout=30).run()

    def test_search_export_state_and_input_invalidation(self):
        self.assertFalse(self.app.exception)
        button = next(b for b in self.app.button if b.label == 'Find candidate processes')
        button.click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.session_state['report']['results'])
        target = self.app.number_input(key='target_UTS')
        target.set_value(1400).run()
        self.assertFalse(self.app.exception)
        self.assertIsNone(self.app.session_state['results'])

    def test_unsupported_family_has_no_inference(self):
        family = next(s for s in self.app.selectbox if s.label == 'Material family')
        family.select('Elastomers').run()
        self.assertFalse(self.app.exception)
        self.assertFalse(any(b.label == 'Find candidate processes' for b in self.app.button))

    def test_forward_prediction(self):
        self.app.radio(key='workspace_mode').set_value('Predict properties').run()
        next(b for b in self.app.button if b.label == 'Predict properties').click().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.error)

    def test_candidate_inspection_keeps_request(self):
        next(b for b in self.app.button if b.label == 'Find candidate processes').click().run()
        report = self.app.session_state['report']
        selector = self.app.selectbox(key='selected_candidate')
        selector.select(selector.options[1]).run()
        self.assertFalse(self.app.exception)
        self.assertEqual(report, self.app.session_state['report'])

    def test_invalid_targets_show_actionable_error(self):
        self.app.number_input(key='target_YS').set_value(1500).run()
        next(b for b in self.app.button if b.label == 'Find candidate processes').click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.error)
        self.assertIsNone(self.app.session_state['results'])

    def test_library_is_not_a_prediction_form(self):
        self.app.radio(key='workspace_mode').set_value('Material library').run()
        self.assertFalse(self.app.exception)
        self.assertFalse(any(b.label == 'Find candidate processes' for b in self.app.button))


if __name__ == '__main__': unittest.main()
