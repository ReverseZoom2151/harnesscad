import unittest

from harnesscad.agents.rag import aql_query as aql


class TestFind(unittest.TestCase):
    def test_novice_query(self):
        q = aql.parse("FIND professors FROM UniversityDW WHERE faculty in the lab")
        self.assertIsInstance(q, aql.FindQuery)
        self.assertEqual(q.columns[0].name, "professors")
        self.assertFalse(q.columns[0].qualified)
        self.assertEqual(q.source.source, "UniversityDW")
        self.assertFalse(q.source.qualified)
        self.assertEqual(q.predicate, "faculty in the lab")

    def test_expert_query(self):
        q = aql.parse("FIND Faculty.name, Faculty.title FROM UniversityDW.Faculty "
                      "WHERE rank in professor")
        self.assertEqual(len(q.columns), 2)
        self.assertTrue(all(c.qualified for c in q.columns))
        self.assertTrue(q.source.qualified)
        self.assertEqual(q.source.relation, "Faculty")

    def test_star(self):
        q = aql.parse("FIND * FROM Emails")
        self.assertTrue(q.star)
        self.assertEqual(q.predicate, "")


class TestJoin(unittest.TestCase):
    def test_join(self):
        text = ("FIND name FROM UniversityDW.Faculty WHERE lab members "
                "JOIN FIND name, award FROM Wikipedia.Pages WHERE turing winners")
        q = aql.parse(text)
        self.assertIsInstance(q, aql.JoinQuery)
        self.assertEqual(q.left.source.source, "UniversityDW")
        self.assertEqual(q.right.columns[1].name, "award")

    def test_double_join_rejected(self):
        with self.assertRaises(aql.ParseError):
            aql.parse("FIND a FROM X JOIN FIND b FROM Y JOIN FIND c FROM Z")


class TestHousekeeping(unittest.TestCase):
    def test_list_sources(self):
        c = aql.parse("?")
        self.assertIsInstance(c, aql.SchemaCommand)
        self.assertEqual(c.target, "")

    def test_list_relations(self):
        c = aql.parse("? UniversityDW")
        self.assertEqual(c.target, "UniversityDW")

    def test_save(self):
        c = aql.parse("SAVE ResearchLabProfessors")
        self.assertIsInstance(c, aql.SaveCommand)
        self.assertEqual(c.name, "ResearchLabProfessors")

    def test_save_requires_name(self):
        with self.assertRaises(aql.ParseError):
            aql.parse("SAVE")


class TestErrorsAndProgram(unittest.TestCase):
    def test_missing_from(self):
        with self.assertRaises(aql.ParseError):
            aql.parse("FIND x WHERE nothing")

    def test_empty(self):
        with self.assertRaises(aql.ParseError):
            aql.parse("   ")

    def test_program(self):
        prog = aql.parse_program("? \n FIND a FROM X \n SAVE r")
        self.assertEqual(len(prog), 3)
        self.assertIsInstance(prog[1], aql.FindQuery)


class TestDelegation(unittest.TestCase):
    def test_expert_delegates_less(self):
        q = aql.parse("FIND Faculty.name FROM DW.Faculty")
        rep = aql.delegation_report(q)
        self.assertEqual(rep["qualified_columns"], 1)
        self.assertEqual(rep["qualified_sources"], 1)
        self.assertEqual(rep["delegated_predicates"], [])

    def test_novice_delegates_predicate(self):
        q = aql.parse("FIND professors FROM DW WHERE the ones in the lab")
        rep = aql.delegation_report(q)
        self.assertEqual(rep["unqualified_columns"], 1)
        self.assertEqual(rep["unqualified_sources"], 1)
        self.assertEqual(rep["delegated_predicates"], ["the ones in the lab"])


if __name__ == "__main__":
    unittest.main()
