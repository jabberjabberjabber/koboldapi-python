import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from koboldapi.chunking.processor import ChunkingProcessor

class TestChunkingProcessor(unittest.TestCase):
    def setUp(self):
        """ Set up test fixtures """
        # Mock KoboldAPI
        self.mock_api = Mock()
        self.mock_api.count_tokens.return_value = {"count": 100}  # Default token count
        self.chunker = ChunkingProcessor(self.mock_api, max_chunk_length=150)

    def test_empty_text(self):
        """ Test chunking empty text returns empty list """
        result = self.chunker.chunk_text("")
        self.assertEqual(result, [])
        self.mock_api.count_tokens.assert_not_called()

    def test_small_text(self):
        """ Test text smaller than chunk size stays whole """
        text = "This is a small piece of text."
        self.mock_api.count_tokens.return_value = {"count": 50}
        
        result = self.chunker.chunk_text(text)
        
        self.assertEqual(len(result), 1)
        chunk, count = result[0]
        self.assertEqual(chunk, text)
        self.assertEqual(count, 50)

    def test_large_text_chunking(self):
        """ Test text larger than chunk size gets split """
        text = ("""We had lain thus in bed, chatting and napping at short intervals, and Queequeg now and then affectionately throwing his brown tattooed legs over mine, and then drawing them back; so entirely sociable and free and easy were we; when, at last, by reason of our confabulations, what little nappishness remained in us altogether departed, and we felt like getting up again, though day-break was yet some way down the future.

Yes, we became very wakeful; so much so that our recumbent position began to grow wearisome, and by little and little we found ourselves sitting up; the clothes well tucked around us, leaning against the head-board with our four knees drawn up close together, and our two noses bending over them, as if our kneepans were warming-pans. We felt very nice and snug, the more so since it was so chilly out of doors; indeed out of bed-clothes too, seeing that there was no fire in the room. The more so, I say, because truly to enjoy bodily warmth, some small part of you must be cold, for there is no quality in this world that is not what it is merely by contrast. Nothing exists in itself. If you flatter yourself that you are all over comfortable, and have been so a long time, then you cannot be said to be comfortable any more. But if, like Queequeg and me in the bed, the tip of your nose or the crown of your head be slightly chilled, why then, indeed, in the general consciousness you feel most delightfully and unmistakably warm. For this reason a sleeping apartment should never be furnished with a fire, which is one of the luxurious discomforts of the rich. For the height of this sort of deliciousness is to have nothing but the blanket between you and your snugness and the cold of the outer air. Then there you lie like the one warm spark in the heart of an arctic crystal.

We had been sitting in this crouching manner for some time, when all at once I thought I would open my eyes; for when between sheets, whether by day or by night, and whether asleep or awake, I have a way of always keeping my eyes shut, in order the more to concentrate the snugness of being in bed. Because no man can ever feel his own identity aright except his eyes be closed; as if darkness were indeed the proper element of our essences, though light be more congenial to our clayey part. Upon opening my eyes then, and coming out of my own pleasant and self-created darkness into the imposed and coarse outer gloom of the unilluminated twelve-o’clock-at-night, I experienced a disagreeable revulsion. Nor did I at all object to the hint from Queequeg that perhaps it were best to strike a light, seeing that we were so wide awake; and besides he felt a strong desire to have a few quiet puffs from his Tomahawk. Be it said, that though I had felt such a strong repugnance to his smoking in the bed the night before, yet see how elastic our stiff prejudices grow when love once comes to bend them. For now I liked nothing better than to have Queequeg smoking by me, even in bed, because he seemed to be full of such serene household joy then. I no more felt unduly concerned for the landlord’s policy of insurance. I was only alive to the condensed confidential comfortableness of sharing a pipe and a blanket with a real friend. With our shaggy jackets drawn about our shoulders, we now passed the Tomahawk from one to the other, till slowly there grew over us a blue hanging tester of smoke, illuminated by the flame of the new-lit lamp.

Whether it was that this undulating tester rolled the savage away to far distant scenes, I know not, but he now spoke of his native island; and, eager to hear his history, I begged him to go on and tell it. He gladly complied. Though at the time I but ill comprehended not a few of his words, yet subsequent disclosures, when I had become more familiar with his broken phraseology, now enable me to present the whole story such as it may prove in the mere skeleton I give.""")
        
        # Mock counts to force chunking
        def token_counter(t):
            if len(t) > 100:  # Full text
                return {"count": 150}
            return {"count": 40}  # Individual sentences
            
        self.mock_api.count_tokens.side_effect = token_counter
        
        result = self.chunker.chunk_text(text)
        
        self.assertGreater(len(result), 1)  # Should have multiple chunks
        total_text = ' '.join(chunk for chunk, _ in result)
        self.assertEqual(total_text.replace('  ', ' ').strip(), text.replace('\n\n', ' ').strip())

    def test_natural_breaks(self):
        """ Test chunking respects sentence boundaries """
        text = "First sentence. Second sentence. Third sentence."
        # Mock token counts to force breaking after second sentence
        counts = {
            text: 120,  # Full text over limit
            "First sentence. ": 30,
            "Second sentence. ": 30,
            "Third sentence.": 30
        }
        self.mock_api.count_tokens.side_effect = lambda t: {"count": counts.get(t, 100)}
        
        result = self.chunker.chunk_text(text)
        
        # Verify chunks break at sentence boundaries
        for chunk, _ in result:
            self.assertTrue(chunk.endswith(". ") or chunk.endswith("."))

    def test_oversized_chunk_handling(self):
        """ Test handling of chunks larger than max size """
        # Single sentence longer than max chunk size
        text = "This_is_a_very_long_sentence_without_breaks."
        self.mock_api.count_tokens.return_value = {"count": 200}  # Over max_chunk_length
        
        result = self.chunker.chunk_text(text)
        
        self.assertEqual(len(result), 1)
        chunk, count = result[0]
        self.assertEqual(chunk, text)  # Keeps oversized chunk rather than breaking words
        self.assertEqual(count, 200)

    @patch('koboldapi.chunking.processor.Extractor')
    def test_chunk_file(self, mock_extractor_class):
        """ Test file chunking with metadata """
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.extract_file_to_string.return_value = (
            "Test content", {"metadata": "test"}
        )
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.set_extract_string_max_length.return_value = mock_extractor
        
        # Test file chunking
        result, metadata = self.chunker.chunk_file("test.txt")
        
        # Verify extractor called correctly
        mock_extractor.extract_file_to_string.assert_called_once_with("test.txt")
        self.assertEqual(metadata, {"metadata": "test"})
        self.assertEqual(len(result), 1)
        
    def test_zero_length_chunk(self):
        """ Test handling of potential zero-length chunks """
        text = "Text with.. Multiple... Dots..."
        
        # Mock to simulate getting a zero-length chunk possibility
        def mock_token_count(t):
            if len(t.strip()) == 0:
                return {"count": 0}
            return {"count": 10}
            
        self.mock_api.count_tokens.side_effect = mock_token_count
        
        result = self.chunker.chunk_text(text)
        
        # Verify no empty chunks in result
        for chunk, count in result:
            self.assertGreater(len(chunk.strip()), 0)
            self.assertGreater(count, 0)

if __name__ == '__main__':
    unittest.main()