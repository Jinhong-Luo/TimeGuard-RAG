import unittest
from timeguard.message_budget import MARKER,pack,shorten_table

class BudgetTest(unittest.TestCase):
    def test_skip_long_and_preserve_fixed_messages(self):
        messages = [{"role":"system","content":"规则"},{"role":"user","content":"问题\n"+MARKER}]
        chunks = [{"content":"过"*60},{"content":"收入12"}]
        pairs, msg = pack(messages,14,chunks,"问题",lambda c,i:f"{i} {c['content']}",len)
        self.assertEqual([c for c,b in pairs],chunks[1:])
        self.assertEqual(msg[-1]["content"],"问题\n1 收入12")

    def test_table_keeps_context(self):
        table = "Income statement (USD millions)\n| Item | 2021 | 2022 |\n| Revenue | 2 | 3 |\n| Costs | 1 | 2 |\nNote 1 includes transfers."
        short = shorten_table(table,"revenue")
        for value in ("Income statement","USD millions","2021","2022","Revenue | 2 | 3","Note 1"):
            self.assertIn(value,short)
        self.assertNotIn("Costs",short)

    def test_no_room_for_instructions(self):
        with self.assertRaisesRegex(ValueError,"预算"):
            pack([{"role":"user","content":"原指令"+MARKER}],2,[],"问题",None,len)
