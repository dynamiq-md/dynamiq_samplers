import dynamiq_engine as dynq
import numpy as np
from dynamiq_samplers.tests.tools import *
# TODO: change this up a little so it isn't import *
from dynamiq_samplers import *


class testInitialConditionSampler(object):
    def setup(self):
        # TOD: clean this up!
        pes = dynq.potentials.interactions.ConstantInteraction(1.0)
        topology = dynq.Topology(np.array([1.0]), pes)
        template = dynq.Snapshot(coordinates=np.array([0.0]),
                                 momenta=np.array([0.0]),
                                 topology=topology)
        self.eng1 = dynq.DynamiqEngine(
            potential=pes, 
            integrator=dynq.integrators.CandyRozmus4(0.1, pes),
            template=template
        )
        self.eng2 = dynq.DynamiqEngine(
            potential=pes, 
            integrator=dynq.integrators.CandyRozmus4(1.0, pes),
            template=template
        )
        # in case other tests have been run before
        if InitialConditionSampler.global_engine is not None:
            InitialConditionSampler.global_engine = None
        self.sampler = InitialConditionSampler()

    @raises(RuntimeError)
    def test_no_initial_engine(self):
        eng = self.sampler.engine

    def test_local_engine_overrides_global(self):
        # set the global engine: this should set the engine
        InitialConditionSampler.global_engine = self.eng1
        assert_equal(self.sampler.engine, self.eng1)
        # set the local engine: this should override the previous
        self.sampler.engine = self.eng2
        assert_equal(self.sampler.engine, self.eng2)

    def test_global_engine_no_override(self):
        # set the local engine: this should set the engine
        self.sampler.engine = self.eng2
        assert_equal(self.sampler.engine, self.eng2)
        # set the global engine: this should not change anything
        InitialConditionSampler.global_engine = self.eng1
        assert_equal(self.sampler.engine, self.eng2)

    def test_global_engine_is_global(self):
        InitialConditionSampler.global_engine = self.eng1
        new_sampler = InitialConditionSampler()
        assert_equal(new_sampler.engine, self.eng1)
        other_class_sampler = OrthogonalInitialConditions([])
        assert_equal(other_class_sampler.engine, self.eng1)

    def test_prepare(self):
        InitialConditionSampler.global_engine = self.eng1
        # prepare without engine
        self.sampler.prepare(n_frames=10)
        assert_equal(self.sampler.n_frames, 10)
        assert_equal(self.sampler.engine, self.eng1)
        # prepare with other engine
        self.sampler.prepare(n_frames=20, engine=self.eng2)
        assert_equal(self.sampler.n_frames, 20)
        assert_equal(self.sampler.engine, self.eng2)


class testOrthogonalInitialConditions(object):
    def setup(self):
        from dynamiq_engine.tests.stubs import PotentialStub
        topology = dynq.Topology(masses=np.array([0.5, 0.5]),
                                 potential=PotentialStub(2))
        self.normal_sampler = GaussianInitialConditions(
            x0=[0.0, 0.0], p0=[0.0, 0.0], 
            alpha_x=[2.0, 1.0], alpha_p=[2.0, 1.0]
        )
        self.e_sampler = MMSTElectronicGaussianInitialConditions.with_n_dofs(2)
        self.sampler = OrthogonalInitialConditions([self.normal_sampler,
                                                    self.e_sampler])
        # TODO set up snapshots
        self.snap0x0 = dynq.MMSTSnapshot(
            coordinates=np.array([0.0, 0.0]),
            momenta=np.array([0.0, 0.0]),
            electronic_coordinates=np.array([0.0, 0.0]),
            electronic_momenta=np.array([0.0, 0.0]),
            topology=topology
        )
        self.snap0x5 = dynq.MMSTSnapshot(
            coordinates=np.array([0.5, 0.0]),
            momenta=np.array([0.5, 0.0]),
            electronic_coordinates=np.array([0.5, 0.0]),
            electronic_momenta=np.array([0.5, 0.0]),
            topology=topology
        )
        self.snap1x0 = dynq.MMSTSnapshot(
            coordinates=np.array([1.0, 0.0]),
            momenta=np.array([1.0, 0.0]),
            electronic_coordinates=np.array([1.0, 0.0]),
            electronic_momenta=np.array([1.0, 0.0]),
            topology=topology
        )

    @raises(RuntimeError)
    def test_error_with_none_dofs_overlap(self):
        part_sampler = GaussianInitialConditions(x0=[0.0], alpha_x=[1.0],
                                                 p0=[], alpha_p=[],
                                                 coordinate_dofs=[1],
                                                 momentum_dofs=[])
        sampler = OrthogonalInitialConditions([self.normal_sampler,
                                               self.e_sampler,
                                               part_sampler])


    @raises(RuntimeError)
    def test_error_with_dofs_overlap(self):
        part_sampler = GaussianInitialConditions(x0=[0.0], alpha_x=[1.0],
                                                 p0=[], alpha_p=[],
                                                 coordinate_dofs=[1],
                                                 momentum_dofs=[])
        part_sampler2 = GaussianInitialConditions(x0=[0.0], alpha_x=[1.0],
                                                  p0=[], alpha_p=[],
                                                  coordinate_dofs=[1],
                                                  momentum_dofs=[])
        sampler = OrthogonalInitialConditions([part_sampler, part_sampler2])


    def test_features(self):
        from openpathsampling.features import coordinates as f_coordinates
        from dynamiq_engine.features import momenta as f_momenta
        from dynamiq_engine.features import electronic_coordinates \
                as f_e_coordinates
        from dynamiq_engine.features import electronic_momenta \
                as f_e_momenta
        
        assert_equal(
            set(self.sampler.__features__), 
            set([f_coordinates, f_momenta, f_e_coordinates, f_e_momenta])
        )
        assert_equal(len(self.sampler.feature_dofs.keys()), 4)
        assert_equal(self.sampler.feature_dofs,
                     {f_coordinates : None, f_momenta : None,
                      f_e_coordinates : None, f_e_momenta : None})

        subsampler1 = MMSTElectronicGaussianInitialConditions(
            x0=[0.0], alpha_x=[1.0], p0=[], alpha_p=[],
            coordinate_dofs=[1], momentum_dofs=[]
        )
        subsampler2 = GaussianInitialConditions(
            x0=[0.0], alpha_x=[1.0], p0=[], alpha_p=[],
            coordinate_dofs=[1], momentum_dofs=[]
        )
        new_sampler = OrthogonalInitialConditions([subsampler1, subsampler2])
        assert_equal(
            set(new_sampler.__features__), 
            set([f_coordinates, f_momenta, f_e_coordinates, f_e_momenta])
        )
        assert_equal(new_sampler.feature_dofs,
                     {f_coordinates : [1], f_momenta : [],
                      f_e_coordinates : [1], f_e_momenta : []})


    def test_sampler(self):
        norm = 0.0205319645093687 # 2.0 / pi^4
        tests = {
            self.snap0x0 : norm,
            self.snap0x5 : norm*np.exp(-2.0*(0.5**2)-4.0*(0.5**2)),
            self.snap1x0 : norm*np.exp(-2.0*(1.0**2)-4.0*(1.0**2)),
        }
        check_function(self.sampler, tests)

        snap = self.sampler.generate_initial_snapshot(self.snap0x0)
        assert_not_equal(snap, self.snap0x0)
        assert_equal(snap.topology, self.snap0x0.topology)
        x_a = list(snap.coordinates.ravel())
        x_b = list(self.snap0x0.coordinates.ravel())
        p_a = list(snap.momenta.ravel())
        p_b = list(self.snap0x0.momenta.ravel())
        x_e_a = list(snap.electronic_coordinates.ravel())
        x_e_b = list(self.snap0x0.electronic_coordinates.ravel())
        p_e_a = list(snap.electronic_momenta.ravel())
        p_e_b = list(self.snap0x0.electronic_momenta.ravel())
        all_a = x_a + p_a + x_e_a + p_e_a
        all_b = x_b + p_b + x_e_b + p_e_b
        for (a, b) in zip(all_a, all_b):
            assert_not_equal(a, b)
