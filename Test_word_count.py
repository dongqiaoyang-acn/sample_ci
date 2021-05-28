import collections
import logging
import re
import tempfile

import unittest
from datetime import datetime as dt


import apache_beam as beam
from apache_beam.coders import VarIntCoder
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.transforms.window import TimestampedValue


class IndexPCollection(beam.DoFn):

    INDEX_STATE = ReadModifyWriteStateSpec(
        name='index', coder=VarIntCoder())

    def process(self, element, index=beam.DoFn.StateParam(INDEX_STATE)):
        current_index = index.read() or 0
        yield current_index, element
        index.write(current_index + 1)


class IndexPCollectionTestCase(unittest.TestCase):

    def test_should_index_non_windowed_p_collection_elements(self):
        with TestPipeline() as p:
            # given
            unindexed_collection = p | 'LoadTestCollection' >> beam.Create([
                ('fruit', 'apple'),
                ('fruit', 'orange'),
                ('fruit', 'watermelon'),
                ('vegetable', 'carrot'),
                ('vegetable', 'potato'),
                ('vegetable', 'broccoli'),
            ])

            # when
            result = unindexed_collection | 'IndexPCollection' >> beam.ParDo(IndexPCollection())

            # then
            assert_that(result, equal_to([
                (0, ('fruit', 'apple')),
                (1, ('fruit', 'orange')),
                (2, ('fruit', 'watermelon')),

                (0, ('vegetable', 'carrot')),
                (1, ('vegetable', 'potato')),
                (2, ('vegetable', 'broccoli')),
            ]))



    def test_should_index_windowed_p_collection_elements(self):
        with TestPipeline() as p:
            # given
            unindexed_collection = p | 'LoadTestCollection' >> beam.Create([
                TimestampedValue(
                    value=('fruit', 'apple'),
                    timestamp=dt.timestamp(dt(2020, 1, 1, hour=1, minute=5))),
                TimestampedValue(
                    value=('fruit', 'orange'),
                    timestamp=dt.timestamp(dt(2020, 1, 1, hour=1, minute=6))),
                TimestampedValue(
                    value=('fruit', 'watermelon'),
                    timestamp=dt.timestamp(dt(2020, 1, 1, hour=1, minute=12))),

                TimestampedValue(
                    value=('vegetable', 'carrot'),
                    timestamp=dt.timestamp(dt(2020, 1, 1, hour=1, minute=5))),
                TimestampedValue(
                    value=('vegetable', 'potato'),
                    timestamp=dt.timestamp(dt(2020, 1, 1, hour=1, minute=6))),
                TimestampedValue(
                    value=('vegetable', 'broccoli'),
                    timestamp=dt.timestamp(dt(2020, 1, 1, hour=1, minute=12))),
            ])

            # when
            result = (unindexed_collection
                      | 'ApplyWindow' >> beam.WindowInto(beam.window.FixedWindows(60 * 10))
                      | 'UnwrappingFromTimestampedValue' >> beam.Map(lambda e: e)
                      | 'IndexPCollection' >> beam.ParDo(IndexPCollection()))

            # then
            assert_that(result, equal_to([
                (0, ('fruit', 'apple')),
                (1, ('fruit', 'orange')),
                (0, ('fruit', 'watermelon')),

                (0, ('vegetable', 'carrot')),
                (1, ('vegetable', 'potato')),
                (0, ('vegetable', 'broccoli')),
            ]))

    def test_count(self):
        with TestPipeline() as p:
            # given
            unindexed_collection = p | 'LoadTestCollection' >> beam.Create([
              "hi", "there", "hi", "hi", "sue", "bob",
              "hi", "sue", "", "", "ZOW", "bob", ""
            ])

            # when
            result = unindexed_collection | 'testData' >> beam.combiners.Count.PerElement();

            # then
            assert_that(result, equal_to([
                ("hi", 4),
                ("there", 1),
                ("sue", 2),
                ("bob", 2),
                ("", 3),
                ("ZOW", 1)]))


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
