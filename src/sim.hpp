#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/rand.hpp>

#include "consts.hpp"
#include "types.hpp"

namespace madEscape {

class Engine;

// This enum is used to uniquely identify each ECS system task graph
// that can be separately executed
enum class TaskGraphID : uint32_t {
  Step,
  NumTaskGraphs,
};

// This enum is used by the Sim and Manager classes to track the export slots
// for each component exported to the training code.
enum class ExportID : uint32_t {
    Reset,
    Action,
    Reward,
    Done,
    SelfObservation,
    PartnerObservations,
    RoomEntityObservations,
    DoorObservation,
    Lidar,
    StepsRemaining,
    NumExports,
};

// Stores values for the ObjectID component that links entities to
// render / physics assets.
enum class SimObject : uint32_t {
    Cube,
    Wall,
    Door,
    Agent,
    Button,
    BasketballHoop,
    Basketball,
    BasketballCourt,
    Plane,
    NumObjects,
};

// The Sim class encapsulates the per-world state of the simulation.
// Sim is always available by calling ctx.data() given a reference
// to the Engine / Context object that is passed to each ECS system.
//
// Per-World state that is frequently accessed but only used by a few
// ECS systems should be put in a singleton component rather than
// in this class in order to ensure efficient access patterns.
struct Sim : public madrona::WorldBase {
    struct Config {
        bool autoReset;
        RandKey initRandKey;
        madrona::phys::ObjectManager *rigidBodyObjMgr;
        const madrona::render::RenderECSBridge *renderBridge;
    };

    // This class would allow per-world custom data to be passed into
    // simulator initialization, but that isn't necessary in this environment
    struct WorldInit {};

    // Sim::registerTypes is called during initialization
    // to register all components & archetypes with the ECS.
    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    // Sim::setupTasks is called during initialization to build
    // the system task graph that will be invoked by the 
    // Manager class (src/mgr.hpp) for each step.
    static void setupTasks(madrona::TaskGraphManager &mgr,
                           const Config &cfg);

    // The constructor is called for each world during initialization.
    // Config is global across all worlds, while WorldInit (src/init.hpp)
    // can contain per-world initialization data, created in (src/mgr.cpp)
    Sim(Engine &ctx,
        const Config &cfg,
        const WorldInit &);

    // The base random key that episode random keys are split off of
    madrona::RandKey initRandKey;

    // Should the environment automatically reset (generate a new episode)
    // at the end of each episode?
    bool autoReset;

    // Are we enabling rendering? (whether with the viewer or not)
    bool enableRender;

    // Current episode within this world
    uint32_t curWorldEpisode;
    // Random number generator state
    madrona::RNG rng;

    // Floor plane entity, constant across all episodes.
    Entity floorPlane;

    // Border wall entities: 3 walls to the left, up and down that define
    // play area. These are constant across all episodes.
    Entity borders[3];

    // Agent entity references. This entities live across all episodes
    // and are just reset to the start of the level on reset.
    Entity agents[consts::numAgents];
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
public:
    using CustomContext::CustomContext;

    // These are convenience helpers for creating renderable
    // entities when rendering isn't necessarily enabled
    template <typename ArchetypeT>
    inline madrona::Entity makeRenderableEntity();
    inline void destroyRenderableEntity(Entity e);
};

}

#include "sim.inl"
