import unittest
from utils.task import extract_task_name

class TestExtractTaskName(unittest.TestCase):
  def test_extract_task_name_valid(self):
    self.assertEqual(extract_task_name("path/to/Task1_Segmentation/file.txt"), "task1_segmentation")
    self.assertEqual(extract_task_name("another/path/Task2_Classification/data.csv"), "task2_classification")
    self.assertEqual(extract_task_name("Task3_Detection/image.png"), "task3_detection")

  def test_extract_task_name_invalid(self):
    self.assertIsNone(extract_task_name("path/to/no_task_here/file.txt"))
    self.assertIsNone(extract_task_name("another/path/without_task/data.csv"))
    self.assertIsNone(extract_task_name("just_a_file.png"))

  def test_extract_task_name_edge_cases(self):
    self.assertEqual(extract_task_name("Task4_Analysis"), "task4_analysis")
    self.assertEqual(extract_task_name("prefix_Task5_Review_suffix"), "task5_review")
    self.assertIsNone(extract_task_name("Task_WithoutNumber"))

if __name__ == "__main__":
  unittest.main()